# MANIFOLD MIXUP: LEARNING BETTER REPRESENTATIONS BY INTERPOLATING HIDDEN STATES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep networks often perform well on the data distribution on which they are trained, yet give incorrect (and often very confident) answers when evaluated on points from off of the training distribution. This is exemplified by the adversarial examples phenomenon but can also be seen in terms of model generalization and domain shift. Ideally, a model would assign lower confidence to points unlike those from the training distribution. We propose a regularizer which addresses this issue by training with interpolated hidden states and encouraging the classifier to be less confident at these points. Because the hidden states are learned, this has an important effect of encouraging the hidden states for a class to be concentrated in such a way so that interpolations within the same class or between two different classes do not intersect with the real data points from other classes. This has a major advantage in that it avoids the underfitting which can result from interpolating in the input space. We prove that the exact condition for this problem of underfitting to be avoided by Manifold Mixup is that the dimensionality of the hidden states exceeds the number of classes, which is often the case in practice. Additionally, this concentration can be seen as making the features in earlier layers more discriminative. We show that despite requiring no significant additional computation, Manifold Mixup achieves large improvements over strong baselines in supervised learning, robustness to single-step adversarial attacks, semi-supervised learning, and Negative Log-Likelihood on held out samples.

# 1 INTRODUCTION

Machine learning systems have been enormously successful in domains such as vision, speech, and language and are now widely used both in research and industry. Modern machine learning systems typically only perform well when evaluated on the same distribution that they were trained on. However machine learning systems are increasingly being deployed in settings where the environment is noisy, subject to domain shifts, or even adversarial attacks. In many cases, deep neural networks which perform extremely well when evaluated on points on the data manifold give incorrect answers when evaluated on points off the training distribution, and with strikingly high confidence.

This manifests itself in several failure cases for deep learning. One is the problem of adversarial examples (Szegedy et al., 2013), in which deep neural networks with nearly perfect test accuracy can produce incorrect classifications with very high confidence when evaluated on data points with small (imperceptible to human vision) adversarial perturbations. These adversarial examples could present serious security risks for machine learning systems. Another failure case involves the training and testing distributions differing significantly. With deep neural networks, this can often result in dramatically reduced performance.

To address these problems, our *Manifold Mixup* approach builds on following assumptions and motivations: (1) we adopt the manifold hypothesis, that is, data is concentrated near a lower-dimensional non-linear manifold (this is the only required assumption on the data generating distribution for *Manifold Mixup* to work); (2) a neural net can learn to transform the data non-linearly so that the transformed data distribution now lies on a nearly flat manifold; (3) as a consequence, linear interpolations between examples in the hidden space also correspond to valid data points, thus providing novel training examples.

![](images/f7206c934567e825c49b5b46b40bfda7f16eee14e732c77af97e4beeb0a9dece.jpg)  
a.

![](images/46280b36f52379e528b0e2458caa24d3cb39dd6656db5a13fff5a9fd4627f824.jpg)  
b.

![](images/cc84bff484810beb7ef884680adb927d55d7872b7c7f1e9a3013b4d5a2962583.jpg)  
C.

![](images/8cb24563788c003ce25662a67f202b1f711709e9ea101eb30cf33c8fe793c720.jpg)  
d.

![](images/3fdc0644db8e3b3572186ab66e833dd8df25ecf8c9e4b00af0e6f200caf39eac.jpg)  
e.

![](images/6e27aa1bbee1f502a3bdc8f98ffd79e0dbf6605805c7d3f26d25533a9c0a49ad.jpg)  
f.  
Figure 1: The top row (a,b,c) shows the decision boundary on the 2d spirals dataset trained with a baseline model (a fully connected neural network with nine layers where middle layer is a 2D bottleneck layer), Input Mixup with  $\alpha = 1.0$ , and *Manifold Mixup* applied only to the 2D bottleneck layer. As seen in (b), Input Mixup can suffer from underfitting since the interpolations between two samples may intersect with a real sample. Whereas *Manifold Mixup* (c), fits the training data perfectly. The bottom row (d,e,f) shows the hidden states for the baseline, Input Mixup, and manifold mixup respectively. *Manifold Mixup* concentrates the labeled points from each class to a very tight region, as predicted by our theory (Section 3) and assigns lower confidence classifications to broad regions in the hidden space. The black points in the bottom row are the hidden states of the points sampled uniformly in x-space and it can be seen that manifold mixup does a better job of giving low confidence to these points. Additional results in Figure 6 of Appendix B show that the way *Manifold Mixup* changes the representations is not accomplished by other well-studied regularizers (weight decay, dropout, batch normalization, and adding noise to the hidden states).

Manifold Mixup performs training on the convex combinations of the hidden state representations of data samples. Previous work, including the study of analogies through word embeddings (e.g. king - man + woman ≈ queen), has shown that such linear interpolation between hidden states is an effective way of combining factors (Mikolov et al., 2013). Combining such factors in the higher level representations has the advantage that it is typically lower dimensional, so a simple procedure like linear interpolation between pairs of data points explores more of the space and with more of the points having meaningful semantics. When we combine the hidden representations of training examples, we also perform the same linear interpolation in the labels (seen as one-hot vectors or categorical distributions), producing new soft targets for the mixed examples.

In practice, deep networks often learn representations such that there are few strong constraints on how the states can be distributed in the hidden space, because of which the states can be widely distributed through the space, (as seen in Figure 1d). As well as, nearly all points in hidden space correspond to high confidence classifications even if they correspond to off-the-training distribution samples (seen as black points in Figure 1d). In contrast, the consequence of our Manifold Mixup approach is that the hidden states from real examples of a particular class are concentrated in local regions and the majority of the hidden space corresponds to lower confidence classifications. This concentration of the hidden states of the examples of a particular class into a local regions enables learning more discriminative features. A low-dimensional example of this can be seen in Figure 1 and a more detailed analytical discussion for what "concentrating into local regions" means is in Section 3.

Our method provides the following contributions:

- The introduction of a novel regularizer which outperforms competitive alternatives such as Cutout (Devries & Taylor, 2017), Mixup (Zhang et al., 2017), AdaMix (Guo et al., 2016), and Dropout (Hinton et al., 2012). On CIFAR-10, this includes a  $50\%$  reduction in test Negative Log-Likelihood (NLL) from 0.1945 to 0.0957.  
- Manifold Mixup achieves significant robustness to single step adversarial attacks.  
- A new method for semi-supervised learning which uses a Manifold Mixup based consistency loss. This method reduces error relative to Virtual Adversarial Training (VAT) (Miyato et al., 2017) by  $21.86\%$  on CIFAR-10, and unlike VAT does not involve any additional significant computation.  
- An analysis of Manifold Mixup and exact sufficient conditions for Manifold Mixup to achieve consistent interpolations. Unlike Input Mixup, this doesn't require strong assumptions about the data distribution (see the failure case of Input Mixup in Figure 1): only that the number of hidden units exceeds the number of classes, which is easily satisfied in many applications.

# 2 MANIFOLD MIXUP

The Manifold Mixup algorithm consists of selecting a random layer (from a set of eligible layers including the input layer)  $k$ . We then process the batch without any mixup until reaching that layer, and we perform mixup at that hidden layer, and then continue processing the network starting from the mixed hidden state, changing the target vector according to the mixup interpolation. More formally, we can redefine our neural network function  $y = f(x)$  in terms of  $k$ :  $f(x) = g_k(h_k(x))$ . Here  $g_k$  is a function which runs a neural network from the input hidden state  $k$  to the output  $y$ , and  $h_k$  is a function which computes the  $k$ -th hidden layer activation from the input  $x$ .

For the linear interpolation between factors, we define a variable  $\lambda$  and we sample from  $p(\lambda)$ . Following (Zhang et al., 2017), we always use a beta distribution  $p(\lambda) = \text{Beta}(\alpha, \alpha)$ . With  $\alpha = 1.0$ , this is equivalent to sampling from  $U(0,1)$ .

We consider interpolation in the set of layers  $S_{k}$  and minimize the expected Manifold Mixup loss.

$$
L = \mathbb {E} _ {\left(x _ {i}, y _ {i}\right), \left(x _ {j}, y _ {j}\right) \sim p (x, y), \lambda \sim p (\lambda), k \sim S _ {k}} \ell \left(f _ {k} \left(\lambda g _ {k} \left(x _ {i}\right) + (1 - \lambda) g _ {k} \left(x _ {j}\right)\right)\right), \lambda y _ {i} + (1 - \lambda) y _ {j}) \tag {1}
$$

We backpropagate gradients through the entire computational graph, including to layers before the mixup process is applied (Section 5.1 and appendix Section B explore this issue directly). In the case where  $k = 0$  is the input layer and  $S_{k} = 0$ , Manifold Mixup reduces to the mixup algorithm of Zhang et al. (2017). With  $\alpha = 2.0$ , about  $5\%$  of the time  $\lambda$  is within  $5\%$  of 0 or 1, which essentially means that an ordinary example is presented. In the more general case, we can optimize the expectation in the Manifold Mixup objective by sampling a different layer to perform mixup in on each update. We could also select a new random layer as well as a new lambda for each example in the minibatch. In theory this should reduce the variance in the updates introduced by these random variables. However in practice we found that this didn't have a significant effect on the results, so we decided to sample a single lambda and a randomly chosen layer per minibatch.

In comparison to Input Mixup, the results in the Figure 2 demonstrate that Manifold Mixup reduces the loss calculated along hidden interpolations significantly better than Input Mixup, without significantly changing the loss calculated along visible space interpolations.

# 3 HOW MANIFOLD MIXUP CHANGES REPRESENTATIONS

Our goal is to show that if one does mixup in a sufficiently deep hidden layer in a deep network, then a mixup loss of zero can be achieved so long the dimensionality of that hidden layer  $\dim (\mathcal{H})$  is greater than the number of classes  $d$ . More specifically the resulting representations for that class must fall onto a subspace of dimension  $\dim (\mathcal{H}) - d$ .

Assume  $\mathcal{X}$  and  $\mathcal{H}$  to denote the input and representation spaces, respectively. We denote the label-set by  $\mathcal{V}$  and let  $\mathcal{Z} \triangleq \mathcal{X} \times \mathcal{Y}$ . Also, let us denote the set of all probability measures on  $\mathcal{Z}$  by  $M(\mathcal{Z})$ . Assume  $\mathcal{G} \subseteq \mathcal{H}^{\mathcal{X}}$  to be the set of all possible functions that can be generated by the neural network

mapping input to the representation space. In this regard, each  $g \in \mathcal{G}$  represents a mapping from input to the representation units. A similar definition can be made for  $\mathcal{F} \subseteq \mathcal{V}^{\mathcal{H}}$ , as the space of all possible functions from the representation space to the output.

We are interested in the solution of the following problem, at least in some specific asymptotic regimes:

$$
J (L, P) \triangleq \inf  _ {g \in \mathcal {G}, f \in \mathcal {F}} \mathbb {E} _ {\lambda} \left\{\int_ {\mathcal {Z} ^ {2}} L \left(f \circ \operatorname {M i x} _ {\lambda} \left(g \left(\boldsymbol {X} _ {1}\right), g \left(\boldsymbol {X} _ {2}\right)\right), \operatorname {M i x} _ {\lambda} \left(\boldsymbol {y} _ {1}, \boldsymbol {y} _ {2}\right)\right) \prod_ {i = 1} ^ {2} \mathrm {d} P \left(\boldsymbol {X} _ {i}, y _ {i}\right) \right\}, \tag {2}
$$

where

$$
\operatorname {M i x} _ {\lambda} (a, b) \triangleq \lambda a + (1 - \lambda) b, \quad \lambda \in [ 0, 1 ], \tag {3}
$$

for any  $a$  and  $b$  defined on the same domain.

We analyze the above-mentioned minimization when the probability measure  $P = \mathbb{P}_D$  is chosen as the empirical distribution over a finite dataset of size  $n$ , denoted by  $D = \{(X_i, y_i)\}_{i=1}^n$ . Let  $f^* \in \mathcal{F}$  and  $g^* \in \mathcal{G}$  be the minimizers in (2) with  $P = \mathbb{P}_D$ .

In particular, we are interested in the case where  $\mathcal{G} = \mathcal{H}^{\mathcal{X}},\mathcal{F} = \mathcal{Y}^{\mathcal{H}}$ , and  $\mathcal{H}$  is a vector space; These conditions simply state that the two respective neural networks which map input into representation space, and representation space to the output are being extended asymptotically<sup>1</sup>. In this regard, we show that the minimizer  $f^{*}$  is a linear function from  $\mathcal{H}$  to  $\mathcal{V}$ . This way, it is easy to show that the following equality holds:

$$
J \left(L, \mathbb {P} _ {\boldsymbol {D}}\right) = \inf  _ {\boldsymbol {h} _ {1}, \dots , \boldsymbol {h} _ {n} \in \mathcal {H}} \frac {1}{n (n - 1)} \sum_ {\substack {i, j = 1 \\ i \neq j}} ^ {n} \left\{\inf  _ {f \in \mathcal {F}} \int_ {0} ^ {1} L \left(f \circ \operatorname {M i x} _ {\lambda} \left(\boldsymbol {h} _ {i}, \boldsymbol {h} _ {j}\right), \operatorname {M i x} _ {\lambda} \left(\boldsymbol {y} _ {i}, \boldsymbol {y} _ {j}\right)\right) \mathrm {d} \lambda \right\}, \tag{4}
$$

where  $\pmb{h}_i \triangleq g(\pmb{X}_i)$  is the representation of  $\pmb{X}_i$ .

Theorem 1. Assume  $\mathcal{H}$  to be a vector space with dimension  $\dim (\mathcal{H})$ , and let  $d\in \mathbb{N}$  to represent the number of distinct classes in dataset  $\pmb{D}$ . Then, if  $\dim (\mathcal{H})\geq d - 1$ ,  $J(L,\mathbb{P}_D) = 0$  and the minimizer function  $f^{*}$  is a linear map from  $\mathcal{H}$  to  $\mathbb{R}^d$ .

Proof. With basic linear algebra, one can confirm that the following argument is true as long as  $\dim(\mathcal{H}) \geq d - 1$ :

$$
\exists \boldsymbol {A}, \boldsymbol {H} \in \mathbb {R} ^ {\dim (\mathcal {H}) \times d}, \boldsymbol {b} \in \mathbb {R} ^ {d} \quad \text {s u c h t h a t} \quad \boldsymbol {A} ^ {T} \boldsymbol {H} + \boldsymbol {b} \mathbf {1} _ {d} ^ {T} = I _ {d \times d}, \tag {5}
$$

where  $I_{d\times d}$  and  $\mathbf{1}_d$  are the  $d$ -dimensional identity matrix and all-one vector, respectively. In fact,  $\pmb{b}\mathbf{1}_d^T$  is a rank-one matrix, while the rank of identity matrix is  $d$ . Therefore,  $\pmb{A}^T\pmb{H}$  only needs to be rank  $d - 1$ .

Let  $f^{*}(\pmb{h}) \triangleq \pmb{A}\pmb{h} + \pmb{b}$ , for all  $\pmb{h} \in \mathcal{H}$ . Also, let  $g^{*}(\pmb{X}_{i}) = h_{\zeta_{i}}$ , where  $h_{i}$  here means the  $i$ th column of matrix  $\pmb{H}$ , and  $\zeta_{i} \in \{1, \dots, d\}$  is the class-index of the  $i$ th sample. We show that such selections will make the objective in (2) equal to zero (which is the minimum possible value). More precisely, the following relations hold:

$$
\begin{array}{l} \frac{1}{n\left(n - 1\right)}\sum_{\substack{i,j = 1\\ i\neq j}}^{n}\left\{\int_{0}^{1}L\left(f^{*}\circ \operatorname{Mix}_{\lambda}\left(g^{*}\left(\boldsymbol{X}_{i}\right),g^{*}\left(\boldsymbol{X}_{j}\right)\right),\operatorname{Mix}_{\lambda}\left(\boldsymbol{y}_{i},\boldsymbol{y}_{j}\right)\right)  \mathrm{d}\lambda \right\} , \\ = \frac{1}{n\left(n - 1\right)}\sum_{\substack{i,j = 1\\ i\neq j}}^{n}\left\{\int_{0}^{1}L\left(\boldsymbol{A}^{T}\left(\lambda \boldsymbol{h}_{\zeta_{i}} + (1 - \lambda)\boldsymbol{h}_{\zeta_{j}}\right) + \boldsymbol {b},\lambda \boldsymbol{y}_{\zeta_{i}} + (1 - \lambda)\boldsymbol{y}_{\zeta_{j}}\right)\mathrm{d}\lambda \right\} , \\ = \frac{1}{n\left(n - 1\right)}\sum_{\substack{i,j = 1\\ i\neq j}}^{n}\left\{\int_{0}^{1}L\left(u\left(\lambda\right),u\left(\lambda\right)\right)  \mathrm{d}\lambda \right\} \\ = 0. \tag {6} \\ \end{array}
$$

The final equality is a direct result of  $\mathbf{A}^T\mathbf{h}_{\zeta_i} + \mathbf{b} = \mathbf{y}_{\zeta_i}$  for  $i = 1,\dots ,n$ .

![](images/e9225ae443af7a5eebf334e1ae7af26463c31868d49de5a54f29eb8fddbefe84.jpg)

Also, it can be shown that as long as  $\dim(\mathcal{H}) > d - 1$ , then data points in the representation space  $\mathcal{H}$  have some degrees of freedom to move independently.

Corollary 1. Consider the setting in Theorem 1, and assume  $\dim(\mathcal{H}) > d - 1$ . Let  $g^{*} \in \mathcal{G}$  to be the true minimizer of (2) for a given dataset  $D$ . Then, data-points in the representation space, i.e.  $g^{*}(X_{i})$ , fall on a  $(\dim(\mathcal{H}) - d + 1)$ -dimensional subspace.

Proof. In the proof of Theorem 1, we have

$$
\boldsymbol {A} ^ {T} \boldsymbol {H} = I _ {d \times d} - \boldsymbol {b 1} _ {d} ^ {T}. \tag {7}
$$

The r.h.s. of (7) can become a rank-  $(d - 1)$  matrix as long as vector  $\pmb{b}$  is chosen properly. Thus,  $\mathbf{A}$  is free to have a null-space of dimension  $\dim (\mathcal{H}) - d + 1$ . This way, one can assign  $g^{*}(\boldsymbol{X}_{i}) = h_{\zeta_{i}} + e_{i}$ , where  $\pmb{h}_{j}$  and  $\zeta_{i}$  (for  $j = 1,\dots ,d$  and  $i = 1,\dots ,n$ ) are defined in the same way as in Theorem 1, and  $\pmb{e}_i$ s can be arbitrary vectors in the null-space of  $\mathbf{A}$ , i.e.  $\pmb{e}_i\in \ker (\pmb {A})$  for all  $i$ .

This result implies that if the Manifold Mixup loss is minimized, then the representation for each class will lie on a subspace of dimension  $\dim (\mathcal{H}) - d + 1$ . In the most extreme case where  $\dim (\mathcal{H}) = d - 1$ , each hidden state from the same class will be driven to a single point, so the change in the hidden states following any direction on the class-conditional manifold will be zero. In the more general case with a larger  $\dim (\mathcal{H})$ , a majority of directions in  $\mathcal{H}$ -space will not change as we move along the class-conditional manifold.

Why are these properties desirable? First, it can be seen as a flattening of the class-conditional manifold which encourages learning effective representations earlier in the network. Second, it means that the region in hidden space occupied by data points from the true manifold has nearly zero measure. So a randomly sampled hidden state within the convex hull spanned by the data is more likely to have a classification score that is not fully confident (non-zero entropy). Thus it encourages the network to learn discriminative features in all layers of the network and to also assign low-confidence classification decisions to broad regions in the hidden space (this can be seen in Figure 1 and Figure 6).

# 4 RELATED WORK

Regularization is a major area of research in machine learning. *Manifold Mixup* closely builds on two threads of research. The first is the idea of linearly interpolating between different randomly drawn examples and similarly interpolating the labels (Zhang et al., 2017; Tokozume et al., 2017). These methods encourage the output of the entire network to change linearly between two randomly drawn training samples, which can result in underfitting. In contrast, for a particular layer at which mixing is done, *Manifold Mixup* allows lower layers to learn more concentrated features in such a way that it makes it easier for the output of the upper layers to change linearly between hidden states of two random samples, achieving better results (section 5.1 and Appendix B).

Another line of research closely related to *Manifold Mixup* involves regularizing deep networks by perturbing the hidden states of the network. These methods include dropout (Hinton et al., 2012), batch normalization (Ioffe & Szegedy, 2015), and the information bottleneck (Alemi et al., 2016). Notably Hinton et al. (2012) and Ioffe & Szegedy (2015) both demonstrated that regularizers already demonstrated to work well in the input space (salt and pepper noise and input normalization respectively) could also be adapted to improve results when applied to the hidden layers of a deep network. We believe that the regularization effect of *Manifold Mixup* would be complementary to that of these algorithms.

Zhao & Cho (2018) explored improving adversarial robustness by classifying points using a function of the nearest neighbors in a fixed feature space. This involved applying mixup between each set of nearest neighbor examples in that feature space. The similarity between Zhao & Cho (2018) and Manifold Mixup is that both consider linear interpolations in hidden states with the same interpolation applied to the labels. However an important difference is that Manifold Mixup backpropagates gradients through the earlier parts of the network (the layers before where mixup is applied) unlike

Table 1: Supervised Classification Results on CIFAR-10 (left) and CIFAR-100 (right). We note significant improvement with Manifold Mixup especially in terms of Negative log-likelihood (NLL). Please refer to Appendix C for details on the implementation of Manifold Mixup and Manifold Mixup All layers.  $\dagger$  and  $\ddagger$  refer to the results reported in (Zhang et al., 2017) and (Guo et al., 2016) respectively.  

<table><tr><td>Model</td><td>Test Error</td><td>Test NLL</td></tr><tr><td colspan="3">PreActResNet18</td></tr><tr><td>No Mixup</td><td>5.12</td><td>0.2646</td></tr><tr><td>Input Mixup (α = 1.0) †</td><td>3.90</td><td>n/a</td></tr><tr><td>AdaMix ‡</td><td>3.52</td><td>n/a</td></tr><tr><td>Input Mixup (α = 1.0)</td><td>3.50</td><td>0.1945</td></tr><tr><td>Manifold Mixup (α = 2.0)</td><td>2.89</td><td>0.1407</td></tr><tr><td colspan="3">PreActResNet152</td></tr><tr><td>No Mixup</td><td>4.20</td><td>0.1994</td></tr><tr><td>Input Mixup (α = 1.0)</td><td>3.15</td><td>0.2312</td></tr><tr><td>Manifold Mixup (α = 2.0)</td><td>2.76</td><td>0.1419</td></tr><tr><td>Manifold Mixup all layers (α = 6.0)</td><td>2.38</td><td>0.0957</td></tr></table>

(a) CIFAR-10

<table><tr><td>Model</td><td>Test Error</td><td>Test NLL</td></tr><tr><td colspan="3">PreActResNet18</td></tr><tr><td>No Mixup †</td><td>25.60</td><td>n/a</td></tr><tr><td>No Mixup</td><td>24.68</td><td>1.284</td></tr><tr><td>Input Mixup (α = 1.0) †</td><td>21.10</td><td>n/a</td></tr><tr><td>Manifold Mixup (α = 2.0)</td><td>21.05</td><td>0.913</td></tr><tr><td colspan="3">PreActResNet34</td></tr><tr><td>Input Mixup (α = 1.0)</td><td>22.79</td><td>1.085</td></tr><tr><td>Manifold Mixup (α = 2.0)</td><td>20.39</td><td>0.930</td></tr></table>

(b) CIFAR-100

Zhao & Cho (2018). As discussed in Section 5.1 and Appendix B this was found to significantly change the learning process.

# 5 EXPERIMENTS

# 5.1 REGULARIZATION ON SUPERVISED LEARNING

We present results on Manifold Mixup based regularization of networks using the PreActResNet architecture (He et al., 2016). We closely followed the procedure of (Zhang et al., 2017) as a way of providing direct comparisons with the Input Mixup algorithm. We used weight decay of 0.0001 and trained with SGD with momentum and multiplied the learning rate by 0.1 at regularly scheduled epochs. These results for CIFAR-10 and CIFAR-100 are in Table 1a and 1b. We also ran experiments where we took PreActResNet34 models trained on the normal CIFAR-100 data and evaluated them on test sets with artificial deformations (shearing, rotation, and zooming) and showed that Manifold Mixup demonstrated significant improvements (Appendix C Table 5), which suggests that Manifold Mixup performs better on the variations in the input space not seen during the training. We also show that the number of epochs needed to reach good results is not significantly affected by using Manifold Mixup in Figure 8.

To better understand why the method works, we performed an experiment where we trained with Manifold Mixup but blocked gradients immediately after the layer where we perform mixup. On CIFAR-10 PreActResNet18, this caused us to achieve  $4.86\%$  test error when trained on 400 epochs and  $4.33\%$  test error when trained on 1200 epochs. This is better than the baseline, but worse than Manifold Mixup or Input Mixup in both cases. Because we randomly select the layer to mix, each layer of the network is still being trained, although not on every update. This demonstrates that the Manifold Mixup method improves results by changing the layers both before and after the mixup operation is applied.

We also compared *Manifold Mixup* against other strong regularizers. We selected the best performing hyperparameters for each of the following models using a validation set. Using each model's best performing hyperparameters, test error averages and standard deviations for five trials (in %) for CIFAR-10 using *PreResNet50* trained for 600 epochs are: vanilla *PreResNet50*  $(4.96 \pm 0.19)$ , Dropout  $(5.09 \pm 0.09)$ , Cutout (Devries & Taylor, 2017)  $(4.77 \pm 0.38)$ , Mixup  $(4.25 \pm 0.11)$  and *Manifold Mixup*  $(3.77 \pm 0.18)$ . This clearly shows that *Manifold Mixup* has strong regularizing

effects. (Note that the results in Table 1 were run for 1200 epochs and thus these results are not directly comparable.)

We also evaluate the quality of the representations learned by Manifold Mixup by applying K-Nearest Neighbour classifier on the feature extracted from the top layer of PreResNet18 for CIFAR-10. We achieved test errors of  $6.09\%$  (Vanilla PreResNet18),  $5.54\%$  (Mixup) and  $5.16\%$  (Manifold Mixup). It suggests that Manifold Mixup helps learning better representations. Further analysis of how Manifold Mixup changes the representations is given in Appendix B

# 5.2 SEMI-SUPERVISED LEARNING

Semi-supervised learning is concerned with building models which can take advantage of both labeled and unlabeled data. It is particularly useful in domains where obtaining labels is challenging, but unlabeled data is plentiful.

The Manifold Mixup approach to semi-supervised learning is closely related to the consistency regularization approach reviewed by Oliver et al. (2018). It involves minimizing loss on labelled samples as well as unlabeled samples by controlling the tradeoff between these two losses via a consistency coefficient. In the Manifold Mixup approach for semi-supervised learning,

Table 2: Results on semi-supervised learning (SSL) on CIFAR-10 (4k labels) and SVHN (1k labels) (in test error %). All results use the same standardized architecture (WideResNet-28-2). Each experiment was run for 5 trials. † refers to the results reported in Oliver et al. (2018)  

<table><tr><td>SSL Approach</td><td>CIFAR-10</td><td>SVHN</td></tr><tr><td>Supervised †</td><td>20.26 ± 0.38</td><td>12.83 ± 0.47</td></tr><tr><td>Mean-Teacher †</td><td>15.87 ± 0.28</td><td>5.65 ± 0.47</td></tr><tr><td>VAT †</td><td>13.86 ± 0.27</td><td>5.63 ± 0.20</td></tr><tr><td>VAT-EM †</td><td>13.13 ± 0.39</td><td>5.35 ± 0.19</td></tr><tr><td>Semi-supervised Input Mixup</td><td>10.71 ± 0.44</td><td>6.54 ± 0.62</td></tr><tr><td>Semi-supervised Manifold Mixup</td><td>10.26 ± 0.32</td><td>5.70 ± 0.48</td></tr></table>

the loss from labeled examples is computed as normal. For computing loss from unlabeled samples, the model's predictions are evaluated on a random batch of unlabeled data points. Then the normal manifold mixup procedure is used, but the targets to be mixed are the soft target outputs from the classifier. The detailed algorithm for both *Manifold Mixup* and Input Mixup with semi-supervised learning are given in appendix D.

Oliver et al. (2018) performed a systematic study of semi-supervised algorithms using a fixed wide resnet architecture "WRN-28-2" (Zagoruyko & Komodakis, 2016). We evaluate Manifold Mixup using this same setup and achieve improvements for CIFAR-10 over the previously best performing algorithm, Virtual Adversarial Training (VAT) (Miyato et al., 2017) and Mean-Teachers (Tarvainen & Valpola, 2017). For SVHN, Manifold Mixup is competitive with VAT and Mean-Teachers. See Table 2. While VAT requires an additional calculation of the gradient and Mean-Teachers requires repeated model parameters averaging, Manifold Mixup requires no additional (non-trivial) computation.

In addition, we also explore the regularization ability of *Manifold Mixup* in a fully-supervised low-data regime by training a PreResnet-152 model on 4000 labeled images from CIFAR-10. We obtained  $13.64\%$  test error which is comparable with the fully-supervised regularized baseline according to results reported in Oliver et al. (2018). Interestingly, we do not use a combination of two powerful regularizers ("Shake-Shake" and "Cut-out") and the more complex ResNext architecture as in Oliver et al. (2018) and still achieve the same level of test accuracy, while doing much better than the fully supervised baseline not regularized with state-of-the-art regularizers  $(20.26\%)$  error.

# 5.3 ADVERSARIAL EXAMPLES

Adversarial examples in some sense are the "worst case" scenario for models failing to perform well when evaluated with data off the manifold<sup>2</sup>. Because Manifold Mixup only considers a sub-

![](images/d1b95945e3dce6200332e620550f14e0b98d2ba7b5ab5122ef62eb7f635452f8.jpg)  
Bas  
#

![](images/22e46d95384b92de803b38a29fbdb90ae44aaf20d27e25cece1f66f334d26d40.jpg)  
Figure 2: Study of test Negative Log-likelihood (NLL) using the interpolated target values (lower is better) on interpolated points under models trained with the baseline, mixup, and Manifold Mixup. Manifold Mixup dramatically improves performance when interpolating in the hidden states, and very slightly reduces performance when interpolating in the visible space. Y-axis denotes NLL and X-axis denotes the interpolation coefficient

![](images/9dee14ce85469e48facf96ff0db14bcfe02eb7ab8edb2dabfd70c3715985f5a5.jpg)

![](images/a9a59e9eee5424b9f886fb91251449b67170cdaed8ccb63b100c7e95c8c3ff10.jpg)  
Trained with Input Mixup  
Trained with Manifold Mixup

![](images/492a8dc576a3631c72ecf0641fb54117e79207153598eb1c54be2e55940165b2.jpg)

set of directions around data points (namely, those corresponding to interpolations), we would not expect the model to be robust to adversarial attacks which can consider any direction within an epsilon-ball of each example. At the same time, *Manifold Mixup* expands the set of points seen during training, so an intriguing hypothesis is that these overlap somewhat with the set of possible adversarial examples, which would force adversarial attacks to consider a wider set of directions, and potentially be more computationally expensive. To explore this we considered the Fast Gradient Sign Method (FGSM, Goodfellow et al., 2014) which only requires a single gradient update and considers a relatively small subset of adversarial directions. The resulting performance of *Manifold Mixup* against FGSM are given in Table 3. A challenge in evaluating adversarial examples comes from the gradient masking problem in which a defense succeeds solely due to reducing the quality of the gradient signal. Athalye et al. (2018) explored this issue in depth and proposed running an unbounded search for a large number of iterations to confirm the quality of the gradient signal. Our *Manifold Mixup* passed this sanity check (see Appendix F). While we found that *Manifold Mixup* greatly improved robustness to the FGSM attack, especially over Input Mixup (Zhang et al., 2017), we found that *Manifold Mixup* did not significantly improve robustness against the stronger iterative projected gradient descent (PGD) attack (Madry et al., 2017).

Table 3: CIFAR-10 Test Accuracy Results on white-box FGSM (Goodfellow et al., 2014) adversarial attack (higher is better) using PreActResNet18 (left). SVHN Test Accuracy Results on white-box FGSM using WideResNet20-10 (Zagoruyko & Komodakis, 2016). Note that our method achieves some degree of adversarial robustness, against the FGSM attack, despite not requiring any additional (significant) computation. † refers to results reported in (Madry et al., 2017)  

<table><tr><td>CIFAR-10 Models</td><td>FGSM</td></tr><tr><td>Adv. Training (PGD 7-step) †</td><td>56.10</td></tr><tr><td>Adversarial Training + Fortified Networks</td><td>81.80</td></tr><tr><td>Baseline (Vanilla Training)</td><td>36.32</td></tr><tr><td>Input Mixup (α = 1.0)</td><td>71.51</td></tr><tr><td>Manifold Mixup (α = 2.0)</td><td>77.50</td></tr><tr><td>CIFAR-100 Models</td><td>FGSM</td></tr><tr><td>Input Mixup (α = 1.0)</td><td>40.7</td></tr><tr><td>Manifold Mixup (α = 2.0)</td><td>44.96</td></tr></table>

<table><tr><td>SVHN Models</td><td>FGSM</td></tr><tr><td>Baseline</td><td>21.49</td></tr><tr><td>Input Mixup</td><td></td></tr><tr><td>(α = 1.0)</td><td>56.98</td></tr><tr><td>Manifold Mixup</td><td></td></tr><tr><td>(α = 2.0)</td><td>65.91</td></tr><tr><td>Adv. Training</td><td></td></tr><tr><td>(PGD 7-step)</td><td>72.80</td></tr></table>

# 6 VISUALIZATION OF INTERPOLATED STATES

An important question is what kinds of feature combinations are being explored when we perform mixup in the hidden layers as opposed to linear interpolation in visible space. To provide a qualitative study of this, we trained a small decoder convnet (with upsampling layers) to predict an image from the *Manifold Mixup* classifier's hidden representation (using a simple squared error loss in the visible space). We then performed mixup on the hidden states between two random examples, and ran this interpolated hidden state through the convnet to get an estimate of what the point would look like in input space. Similarly to earlier results on auto-encoders (Bengio et al., 2013), we found that these interpolated  $h$  points corresponded to images with a blend of the features from the two images, as opposed to the less-semantic pixel-wise blending resulting from Input Mixup as shown in Figure 3 and Figure 4. Furthermore, this justifies the training objective for examples mixed-up in the hidden layers: (1) most of the interpolated points correspond to combinations of semantically meaningful factors, thus leading to the more training samples; and (2) none of the interpolated points between objects of two different categories A and B correspond to a third category C, thus justifying a training target which gives 0 probability on all the classes except A and B.

![](images/81a43df5d26234376420c10f7223236c50d5d6b0c4bb24323925eb35be3153b0.jpg)

![](images/7ad12e2aaa33b7579bde80ca8634d6423a761fd99446f0614dff35d6918b8ee9.jpg)  
Figure 3: Interpolations in the input space with a mixing rate varied from 0.0 to 1.0.  
Figure 4: Interpolations in the hidden states (using a small convolutional network trained to predict the input from the output of the second resblock). The interpolations in the hidden states show a better blending of semantically relevant features, and more of the images are visually consistent.

# 7 CONCLUSION

Deep neural networks often give incorrect yet extremely confident predictions on data points which are unlike those seen during training. This problem is one of the most central challenges in deep learning both in theory and in practice. We have investigated this from the perspective of the representations learned by deep networks. In general, deep neural networks can learn representations such that real data points are widely distributed through the space and most of the area corresponds to high confidence classifications. This has major downsides in that it may be too easy for the network to provide high confidence classification on points which are off of the data manifold and also that it may not provide enough incentive for the network to learn highly discriminative representations. We have presented *Manifold Mixup*, a new algorithm which aims to improve the representations learned by deep networks by encouraging most of the hidden space to correspond to low confidence

classifications while concentrating the hidden states for real examples onto a lower dimensional subspace. We applied Manifold Mixup to several tasks and demonstrated improved test accuracy and dramatically improved test likelihood on classification, better robustness to adversarial examples from FGSM attack, and improved semi-supervised learning. *Manifold Mixup* incurs virtually no additional computational cost, making it appealing for practitioners.

# REFERENCES

Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information bottleneck. CoRR, abs/1612.00410, 2016. URL http://arxiv.org/abs/1612.00410.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International Conference on Machine Learning, pp. 214-223, 2017.  
A. Athalye, N. Carlini, and D. Wagner. Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. *ArXiv e-prints*, February 2018.  
Yoshua Bengio, Grégoire Mesnil, Yann Dauphin, and Salah Rifai. Better mixing via deep representations. In ICML'2013, 2013.  
Terrance Devries and Graham W. Taylor. Improved regularization of convolutional neural networks with cutout. CoRR, abs/1708.04552, 2017. URL http://arxiv.org/abs/1708.04552.  
J. Gilmer, L. Metz, F. Faghri, S. S. Schoenholz, M. Raghu, M. Wattenberg, and I. Goodfellow. Adversarial Spheres. ArXiv e-prints, January 2018.  
I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and Harnessing Adversarial Examples. ArXiv e-prints, December 2014.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pp. 5769-5779, 2017.  
Hongyu Guo, Yongyi Mao, and Richong Zhang. MixUp as Locally Linear Out-Of-Manifold Regularization. ArXiv e-prints, 2016. URL https://arxiv.org/abs/1809.02499.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. CoRR, abs/1603.05027, 2016. URL http://arxiv.org/abs/1603.05027.  
Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. CoRR, abs/1207.0580, 2012. URL http://arxiv.org/abs/1207.0580.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. CoRR, abs/1502.03167, 2015. URL http://arxiv.org/abs/1502.03167.  
A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards Deep Learning Models Resistant to Adversarial Attacks. ArXiv e-prints, June 2017.  
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013. URL http://arxiv.org/abs/1301.3781.  
T. Miyato, S.-i. Maeda, M. Koyama, and S. Ishii. Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning. ArXiv e-prints, April 2017.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. CoRR, abs/1802.05957, 2018. URL http://arxiv.org/abs/1802.05957.  
A. Oliver, A. Odena, C. Raffel, E. D. Cubuk, and I. J. Goodfellow. Realistic Evaluation of Deep Semi-Supervised Learning Algorithms. ArXiv e-prints, April 2018.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2234-2242, 2016.  
C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. ArXiv e-prints, December 2013.  
Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30, pp. 1195–1204. Curran Associates, Inc., 2017.  
Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada. Between-class learning for image classification. CoRR, abs/1711.10284, 2017. URL http://arxiv.org/abs/1711.10284.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. CoRR, abs/1605.07146, 2016. URL http://arxiv.org/abs/1605.07146.

Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. CoRR, abs/1710.09412, 2017. URL http://arxiv.org/abs/1710.09412.  
Jake Zhao and Kyunghyun Cho. Retrieval-augmented convolutional neural networks for improved robustness against adversarial examples. CoRR, abs/1802.09502, 2018. URL http://arxiv.org/abs/1802.09502.
