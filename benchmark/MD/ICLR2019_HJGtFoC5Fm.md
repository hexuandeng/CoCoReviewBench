# ON THE MARGIN THEORY OF FEEDFORWARD NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Past works have shown that, somewhat surprisingly, over-parametrization can help generalization in neural networks. Towards explaining this phenomenon, we adopt a margin-based perspective. We establish: 1) for multi-layer feedforward relu networks, the global minimizer of a weakly-regularized cross-entropy loss has the maximum normalized margin among all networks, 2) as a result, increasing the over-parametrization improves the normalized margin and generalization error bounds for two-layer networks. In particular, an infinite-size neural network enjoys the best generalization guarantees. The typical infinite feature methods are kernel methods; we compare the neural net margin with that of kernel methods and construct natural instances where kernel methods have much weaker generalization guarantees. We validate this gap between the two approaches empirically. Finally, this infinite-neuron viewpoint is also fruitful for analyzing optimization. We show that a perturbed gradient flow on infinite-size networks finds a global optimizer in polynomial time.

# 1 INTRODUCTION

In deep learning, over-parametrization refers to the widely-adopted technique of using more parameters than necessary (Krizhevsky et al., 2012; Livni et al., 2014). Both computationally and statistically, over-parametrization is crucial for learning neural nets. Controlled experiments demonstrate that over-parametrization eases optimization by smoothing the non-convex loss surface (Livni et al., 2014; Sagun et al., 2017). Statistically, increasing model size without any regularization still improves generalization even after the model interpolates the data perfectly (Neyshabur et al., 2017b). This is surprising given the conventional wisdom on the trade-off between model capacity and generalization.

In the absence of an explicit regularizer, algorithmic regularization is likely the key contributor to good generalization. Recent works have shown that gradient descent finds the minimum norm solution fitting the data for problems including logistic regression, linearized neural networks, and matrix factorization (Soudry et al., 2018; Gunasekar et al., 2018b; Li et al., 2018; Gunasekar et al., 2018a; Ji & Telgarsky, 2018). Many of these proofs require a delicate analysis of the algorithm's dynamics, and some are not fully rigorous due to assumptions on the iterates. To the best of our knowledge, it is an open question to prove analogous results for even two-layer relu networks. (For example, the technique of Li et al. (2018) on two-layer neural nets with quadratic activations still falls within the realm of linear algebraic tools, which apparently do not suffice for other activations.)

We propose a different route towards understanding generalization: making the regularization explicit. The motivations are: 1) with an explicit regularizer, we can analyze generalization without fully understanding optimization; 2) it is unknown whether gradient descent provides additional implicit regularization beyond what  $\ell_2$  regularization already offers; 3) on the other hand, with a sufficiently weak  $\ell_2$  regularizer, we can prove stronger results that apply to multi-layer neural nets with relu activations. Additionally, explicit regularization is perhaps more relevant because  $\ell_2$  regularization is typically used in practice.

Concretely, we add a norm-based regularizer to the cross entropy loss of a multi-layer feedforward neural network with relu activations. We show that the global minimizer of the regularized objective achieves the maximum normalized margin among all the models with the same architecture, if the regularizer is sufficiently weak (Theorem 2.1). Informally, for models with norm 1 that perfectly classify the data, the margin is the smallest difference across all datapoints between the classifier

score for the true label and the next best score. We are interested in normalized margin because its inverse bounds the generalization error (see recent work (Bartlett et al., 2017; Neyshabur et al., 2017a; 2018) and our Theorem 3.1). Our work explains why optimizing the training loss can lead to parameters with a large margin and thus, better generalization error.

At a first glance, it might seem counterintuitive that decreasing the regularizer is the right approach. At a high level, we show that the regularizer only serves as a tiebreaker to steer the model towards choosing the largest normalized margin. Our proofs are simple, oblivious to the optimization procedure, and apply to any norm-based regularizer. We also show that an exact global minimum is unnecessary: if we approximate the minimum loss within a constant, we obtain the max-margin within a constant (Theorem 2.2).

We further study the margin of two-layer networks: let  $\gamma^{\star ,m}$  be the max normalized margin of a neural net with  $m$  hidden units (formally defined in Section 3.1). Let  $\gamma^{\star ,\infty}\triangleq \sup_{m}\gamma^{\star ,m}$  be the largest possible margin of an infinite two-layer network. We will show three properties of the margins:

1. In Theorem 3.2, we show that the optimal normalized margin of two-layer networks is non-decreasing as the width of the architecture grows, so the generalization error bound only improves with a wider network. Thus, even if the dataset is already separable, it could still be useful to increase the width to achieve larger margin and better generalization. More formally, let  $n$  be the number of training examples. We additionally approach the maximum possible margin  $\gamma^{\star,\infty}$  after over-parameterizing with  $m \geq n$  neurons:  $\forall m \geq n, \gamma^{\star,m} = \gamma^{\star,\infty}$ .  
2. The max-margin of infinite-size nets,  $\gamma^{\star,\infty}$ , equals half the margin of the  $\ell_1$ -norm SVM (Zhu et al., 2004) over the lifted feature space defined by the activation function applied to all possible hidden units. (See Theorem 3.3.)  
3. We compare the neural net margin  $\gamma^{\star, \infty}$  to the standard margin for the kernel SVM on the same features. We design a simple data distribution (Figure 1) where neural net margin  $\gamma^{\star, \infty}$  is large but the kernel margin is small. This translates to an  $\Omega(\sqrt{d})$  factor gap between the generalization error bounds for the two approaches and demonstrates the power of neural nets compared to kernel methods. We experimentally confirm that a gap does indeed exist.

In the context of bullet 2, our work is closely related to that of Rosset et al. (2007) and Neyshabur et al. (2014), who show that optimizing the loss over the parameters of a two-layer relu network is equivalent to optimizing the loss of a "convex neural net" parametrized by a distribution over hidden units. We go one step further and connect the weakly regularized training loss to the  $\ell_1$  SVM.

We will also adopt this view of infinite-size neural networks to study how over-parametrization helps optimization. Prior works (Mei et al., 2018; Chizat & Bach, 2018) show that gradient descent on two-layer networks becomes Wasserstein gradient flow over parameter distributions in the limit of infinite neurons. For this setting, we prove that perturbed Wasserstein gradient flow finds a global optimizer in polynomial time.

Finally, we empirically validate several of the claims made in this paper. First, we train a two-layer network on a one-dimensional classification task that is simple to visualize. In one dimension, it is possible to brute-force approximate the maximum neural network margin and we show that training with an progressively smaller regularizer results in convergence to this margin. Second, we compare the generalization performance of neural networks and kernel methods and confirm that neural networks do achieve better generalization, as our theory predicts.

# 1.1 ADDITIONAL RELATED WORK

Zhang et al. (2016) and Neyshabur et al. (2017b) show that neural network generalization defies conventional explanations and requires new ones. One proposed explanation is the inductive bias of the training algorithm. Recent papers (Hardt et al., 2015; Brutzkus et al., 2017; Chaudhari et al., 2016) study inductive bias through training time and sharpness of local minima. Neyshabur et al. (2015a) propose a new steepest descent algorithm in a geometry invariant to weight rescaling and show that this improves generalization. Morcos et al. (2018) relate generalization in deep nets to the number of "directions" in the neurons. Other papers (Gunasekar et al., 2017; Soudry et al., 2018; Gunasekar et al., 2018b; Li et al., 2018; Gunasekar et al., 2018a) study implicit regularization towards a specific solution. Ma et al. (2017) show that implicit regularization can help gradient descent avoid

overshooting optima. Rosset et al. (2004) study logistic regression with a weak regularization and show convergence to the max margin solution. We adopt their techniques and extend their results.

Recent works have also derived tighter Rademacher complexity bounds for deep neural networks (Neyshabur et al., 2015b; Bartlett et al., 2017; Neyshabur et al., 2017a; Golowich et al., 2017) and new compression based generalization properties (Arora et al., 2018b). Dziugaite & Roy (2017) manage to compute non-vacuous generalization bounds from PAC-Bayes bounds. Neyshabur et al. (2018) investigate the Rademacher complexity of two-layer networks. Liang & Rakhlin (2018) and Belkin et al. (2018) study the generalization of kernel methods.

On the optimization side, Soudry & Carmon (2016) explain why over-parametrization can remove bad local minima. Safran & Shamir (2016) show that over-parametrization can improve the quality of the random initialization. Haeffele & Vidal (2015), Nguyen & Hein (2017), and Venturi et al. (2018) show that for sufficiently overparametrized networks, all local minima are global, but do not show how to find these minima via gradient descent. Du & Lee (2018) show that for two-layer networks with quadratic activations, all second-order stationary points are global minimizers. Arora et al. (2018a) interpret over-parametrization as a means of implicit acceleration during optimization. Mei et al. (2018), Chizat & Bach (2018), and Sirignano & Spiliopoulos (2018) take a distributional view of over-parametrized networks. Chizat & Bach (2018) show that Wasserstein gradient flow converges to global optimizers under structural assumptions. We extend this to a polynomial-time result.

# 1.2 NOTATION

Let  $\mathbb{R}$  denote the set of real numbers. We will use  $\| \cdot \|$  to indicate a general norm, with  $\| \cdot \|_1$ ,  $\| \cdot \|_2$ ,  $\| \cdot \|_{\infty}$  denoting the  $\ell_1, \ell_2, \ell_{\infty}$  norms on finite dimensional vectors, respectively, and  $\| \cdot \|_F$  denoting the Frobenius norm on a matrix. In general, we use  $-$  on top of a symbol to denote a unit vector: when applicable,  $\bar{u} \triangleq u / \| u\|$ , where the norm  $\| \cdot \|$  will be clear from context. Let  $\mathbb{S}^{d-1} \triangleq \{\bar{u} \in \mathbb{R}^d : \| \bar{u} \|_2 = 1\}$  be the unit sphere in  $d$  dimensions. Let  $\mathcal{L}^p(\mathbb{S}^{d-1})$  be the space of functions on  $\mathbb{S}^{d-1}$  for which the  $p$ -th power of the absolute value is Lebesgue integrable. For  $\alpha \in \mathcal{L}^p(\mathbb{S}^{d-1})$ , we overload notation and write  $\| \alpha \|_p \triangleq \left( \int_{\mathbb{S}^{d-1}} |\alpha(\bar{u})|^p d\bar{u} \right)^{1/p}$ . Additionally, for  $\alpha_1 \in \mathcal{L}^1(\mathbb{S}^{d-1})$  and  $\alpha_2 \in \mathcal{L}^\infty(\mathbb{S}^{d-1})$  or  $\alpha_1, \alpha_2 \in \mathcal{L}^2(\mathbb{S}^{d-1})$ , we can define  $\langle \alpha_1, \alpha_2 \rangle \triangleq \int_{\mathbb{S}^{d-1}} \alpha_1(\bar{u}) \alpha_2(\bar{u}) d\bar{u} < \infty$ . Furthermore, we will use  $\mathrm{Vol}(\mathbb{S}^{d-1}) \triangleq \int_{\mathbb{S}^{d-1}} 1 d\bar{u}$ .

Throughout this paper, we reserve the symbol  $X = [x_{1},\ldots ,x_{n}]$  to denote the collection of datapoints (as a matrix), and  $Y = [y_{1},\dots,y_{n}]$  to denote labels. We use  $d$  to denote the dimension of our data. We often use  $\Theta$  to denote the parameters of a prediction function  $f$ , and  $f(\Theta ;x)$  to denote the prediction of  $f$  on datapoint  $x$ .

We will use the notation  $\lesssim, \gtrsim$  to mean less than or greater than up to a universal constant, respectively. Unless stated otherwise, we use  $O(\cdot), \Omega(\cdot)$  as a placeholders for some universal constant in upper and lower bounds, respectively. We will use poly to denote some universal constant-degree polynomial in the arguments.

# 2 WEAK REGULARIZER GUARANTEES MAX MARGIN SOLUTIONS

In this section, we will show that when we add a weak regularizer to cross-entropy loss with a positive-homogeneous prediction function, the normalized margin of the optimum converges to some max-margin solution. As a concrete example, feedforward relu networks are positive-homogeneous.

Let  $l$  be the number of labels, so the  $i$ -th example has label  $y_{i} \in [l]$ . We work with a family  $\mathcal{F}$  of prediction functions  $f(\Theta; \cdot): \mathbb{R}^d \to \mathbb{R}^l$  that are  $a$ -positive-homogeneous in their parameters for some  $a > 0$ :  $f(c\Theta; x) = c^{a}f(\Theta; x), \forall c > 0$ . We additionally require that  $f$  is continuous in  $\Theta$ . For some general norm  $\| \cdot \|$ , we study the  $\lambda$ -regularized cross-entropy loss  $L_{\lambda}$ , defined as

$$
L _ {\lambda} (\Theta) \triangleq \sum_ {i = 1} ^ {n} - \log \frac {\exp \left(f _ {y _ {i}} \left(\Theta ; x _ {i}\right)\right)}{\sum_ {j = 1} ^ {l} \exp \left(f _ {j} \left(\Theta ; x _ {i}\right)\right)} + \lambda \| \Theta \| ^ {r} \tag {2.1}
$$

for fixed  $r > 0$ . Let  $\Theta_{\lambda} \in \arg \min L_{\lambda}(\Theta)$ .<sup>1</sup> We define the normalized margin of  $\Theta_{\lambda}$  as:

$$
\gamma_ {\lambda} \triangleq \min  _ {i} \left(f _ {y _ {i}} \left(\bar {\Theta} _ {\lambda}; x _ {i}\right) - \max  _ {j \neq y _ {i}} f _ {j} \left(\bar {\Theta} _ {\lambda}; x _ {i}\right)\right) \tag {2.2}
$$

Define the  $\| \cdot \|$  -max normalized margin as

$$
\gamma^ {\star} \triangleq \max  _ {| | \Theta | | \leq 1} \left[ \min  _ {i} \left(f _ {y _ {i}} (\Theta ; x _ {i}) - \max  _ {j \neq y _ {i}} f _ {j} (\Theta ; x _ {i})\right) \right]
$$

and let  $\Theta^{\star}$  be a parameter achieving this maximum. We show that with sufficiently small regularization level  $\lambda$ , the normalized margin  $\gamma_{\lambda}$  approaches the maximum margin  $\gamma^{\star}$ .

Theorem 2.1. Assume the training data is separable by a network  $f(\Theta^{\star};\cdot) \in \mathcal{F}$  with an optimal normalized margin  $\gamma^{\star} > 0$ . Then, the normalized margin of the global optimum of the weakly-regularized objective (equation 2.1) converges to  $\gamma^{\star}$  as the strength of the regularizer goes to zero. Mathematically, let  $\gamma_{\lambda}$  be defined in equation 2.2. Then

$$
\gamma_ {\lambda} \rightarrow \gamma^ {\star} \text {a s} \lambda \rightarrow 0
$$

An intuitive explanation for our result is as follows: because of the homogeneity, the loss  $L(\Theta_{\lambda})$  roughly satisfies the following (for small  $\lambda$ , and ignoring problem parameters such as  $n$ ):

$$
L _ {\lambda} (\Theta_ {\lambda}) \approx \exp (- \| \Theta_ {\lambda} \| ^ {a} \gamma_ {\lambda}) + \lambda \| \Theta_ {\lambda} \| ^ {r}
$$

Thus, the loss focuses on choosing parameters with larger margin, and the regularization term biases the loss to select parameters with a smaller norm. The full proof of the theorem is deferred to Section A.1.

We can also provide an analogue of Theorem 2.1 for the binary classification setting. For this setting, our prediction is now a single real output and we train using logistic loss. We provide formal definitions and results in Section A.2. Our theory for two-layer neural networks (see Section 3) is based in this setting.

# 2.1 OPTIMIZATION ACCURACY

Since  $L_{\lambda}$  is typically hard to optimize exactly for neural nets, it would be ideal to relax the condition that  $\Theta_{\lambda}$  minimizes  $L_{\lambda}$ . Thus, we ask, how accurately do we need to optimize  $L_{\lambda}$  to obtain a margin that approximates  $\gamma^{\star}$  up to a constant? The following theorem shows that if suffices to find  $\Theta'$  achieving a constant factor multiplicative approximation of  $L_{\lambda}(\Theta_{\lambda})$ , where  $\lambda$  is some sufficiently small polynomial in  $n, l, \gamma^{\star}$ . Though our theorem is stated for the general multi-class setting, our result applies for binary classification as well. We provide the proof in Section A.3.

Theorem 2.2. In the setting of Theorem 2.1, suppose that we choose  $\lambda = \frac{(\gamma^{\star})^{r / a}}{n^{c}(l - 1)^{c}}$  for sufficiently large  $c$  (that only depends on  $r / a$ ). Let  $\Theta'$  denote a 2-approximate minimizer of  $L_{\lambda}$ , so  $L_{\lambda}(\Theta') \leq 2L_{\lambda}(\Theta_{\lambda})$ . Denote the normalized margin of  $\Theta'$  by  $\gamma'$ . Then

$$
\gamma^ {\prime} \geq \frac {\gamma^ {\star}}{2 \cdot 4 ^ {a / r}}
$$

# 3 MARGINS OF OVER-PARAMETERIZED TWO-LAYER HOMOGENEOUS NEURAL NETS

In Section 2 we showed that a weakly-regularized logistic loss leads to the maximum normalized margin. In this section, we analyze the properties of the max-margin of neural nets more closely. We will contrast neural networks with kernel methods, for which margins have already been extensively studied. Towards a first-cut understanding, we focus on two-layer networks for binary classification.

First, in Section 3.1 we provide a bound stating that the generalization error is roughly linear in the inverse of the margin, establishing that a larger margin implies better generalization. In Section 3.2, we show that the maximum normalized margin is non-decreasing with the hidden layer size and stays constant as soon as there are more hidden units than data points. This suggests that increasing the size of the network improves the generalization of the solution.

Second, in Section 3.3, we draw an analogy to classical kernel methods by proving that the maximum  $\ell_2$ -normalized margin of an over-parameterized neural net is equal to half the maximum possible  $\ell_1$ -normalized margin of linear functionals on a lifted feature space. In other words, we establish an equivalence between neural networks and the 1-norm SVM (Zhu et al., 2004) on the lifted features. These features are constructed by applying the activation function on all possible hidden layer weights.

Third, continuing this analogy, we will compare the generalization power of a two-layer neural network to that of a kernel method on the lifted space. This kernel method corresponds to fixing random weights for the hidden layer and solving a 2-norm max-margin problem on the top layer weights. We demonstrate instances where two layer neural networks give better generalization error guarantees than the kernel method.

# 3.1 SETUP AND MARGIN-BASED GENERALIZATION ERROR

In the rest of the paper, we work with two-layer neural networks with a single output for binary classification. We use  $m$  to denote the number of hidden units,  $w_{1},\ldots ,w_{m}\in \mathbb{R}^{d}$  for the weight vectors on the first layer, and  $u_{1},\dots,u_{m}\in \mathbb{R}$  for the weights on the second layer. We let  $\theta_j\triangleq (w_j,u_j)$ , and we use  $\Theta$  to denote the collection of all the parameters. We assume in this section that the activation  $\phi :\mathbb{R}\to \mathbb{R}$  is 1-homogeneous and 1-Lipschitz. The network thus computes a single score

$$
f (\Theta ; x) \triangleq \sum_ {j = 1} ^ {m} w _ {j} \phi (u _ {j} ^ {\top} x)
$$

We consider  $\ell_2$  regularization from here on. The regularized logistic loss of the architecture with  $m$  hidden units is therefore

$$
L _ {\lambda , m} \triangleq \frac {1}{n} \sum_ {i = 1} ^ {n} \log \left(1 + \exp \left(- y _ {i} f (\Theta ; x _ {i})\right)\right) + \lambda \| \Theta \| _ {2} ^ {2} \tag {3.1}
$$

where  $\| \Theta \|_2$  denotes the Euclidean norm of all the parameters in  $\Theta$ . We note that  $f$  and the regularizer are both 2-homogeneous in  $\Theta$ , so the results of Section 2 apply to  $L_{\lambda, m}$ .<sup>3</sup>

Following our conventions from Section 2, we denote the optimizer of  $L_{\lambda,m}$  by  $\Theta_{\lambda,m}$ , the normalized margin of  $\Theta_{\lambda,m}$  by  $\gamma_{\lambda,m}$ , the max-margin solution by  $\Theta^{\star,m}$ , and the max-margin by  $\gamma^{\star,m}$ . We emphasize the size of the network in our notation. Since our classifier  $f$  now predicts a single real value, we need to redefine

$$
\gamma_ {\lambda , m} \triangleq \min  _ {i} y _ {i} f (\bar {\Theta} _ {\lambda , m}; x _ {i})
$$

$$
\gamma^ {\star , m} \triangleq \max  _ {\| \Theta \| \leq 1} \min  _ {i} y _ {i} f (\Theta ; x _ {i})
$$

When the data is not separable by a  $m$ -unit neural net,  $\gamma^{\star,m}$  is zero by definition.

Recall that  $X = [x_{1}, \ldots, x_{n}]$  denotes the matrix with all the data points as columns, and  $Y = [y_{1}, \ldots, y_{n}]$  denotes the labels. We sample  $X$  and  $Y$  i.i.d. from the data generating distribution  $p_{\mathrm{data}}$ , which is supported on  $\mathcal{X} \times \{-1, +1\}$ . We can define the population 0-1 loss and the training 0-1 loss of the network  $\Theta$  as

$$
L(\Theta) = \operatorname *{Pr}_{(x,y)\sim p_{\mathrm{data}}}[yf(\Theta ;x)\leq 0]
$$

We will let  $D \triangleq \frac{\|X\|_F^2}{n}$  be the average norm squared of the data and  $C \triangleq \sup_{x \in \mathcal{X}} \|x\|_2$  be an upper bound on the norm of a single datapoint. The following theorem shows that the generalization error only depends on the parameters through the inverse of the margin on the training data. We provide a proof in Section C.1.

Theorem 3.1. Suppose  $\phi$  is 1-Lipschitz and 1-homogeneous. Then for any  $\Theta$  that separates the data with margin  $\gamma \triangleq \min_{i}y_{i}f(\bar{\Theta};x_{i}) > 0$ , with probability at least  $1 - \delta$  over the draw of  $X,Y$ ,

$$
L (\Theta) \leq \frac {6}{\gamma} \sqrt {\frac {D}{n}} + \epsilon (\gamma) \tag {3.2}
$$

where  $\epsilon (\gamma)\triangleq \sqrt{\frac{\log\log_2\frac{4C}{\gamma}}{n}} +\sqrt{\frac{\log(1 / \delta)}{2n}}$  Note that  $\epsilon (\gamma)$  is typically small, and thus the above bound mainly scales with  $\frac{1}{\gamma}\sqrt{\frac{D}{n}}$  . As a corollary, with probability  $1 - \delta ,^4$

$$
\lim  _ {\lambda \rightarrow 0} L (\Theta_ {\lambda , m}) \leq \frac {6}{\gamma^ {\star , m}} \sqrt {\frac {D}{n}} + \epsilon \left(\gamma^ {\star , m}\right) \tag {3.3}
$$

Above we implicitly assume  $\gamma^{\star ,m} > 0$ , since otherwise the right hand side of the bound is vacuous.

One consequence of the above theorem and Theorem 2.2 is that if  $\lambda$  is polynomially small in  $\gamma^{\star ,m}$  and  $n$ , we only need to optimize  $L_{\lambda ,m}$  up to a constant multiplicative factor to obtain parameters with generalization bounds roughly as good as those for  $\Theta^{\star ,m}$ .

# 3.2 THE MAX MARGIN IS NON-DECREASING IN THE HIDDEN LAYER SIZE

Now we show that the maximum normalized margin is nondecreasing with the hidden layer size and stays constant once we have more hidden units than examples.

Theorem 3.2. In the setting of Section 3.1, recall that  $\gamma^{\star,m}$  denotes the max normalized margin of a two-layer neural network with hidden layer size  $m$ . Then,

$$
\gamma^ {\star , 1} \leq \gamma^ {\star , 2} \dots \leq \gamma^ {\star , n} = \gamma^ {\star , n + 1} = \gamma^ {\star , n + 2} = \dots \tag {3.4}
$$

We note that  $\gamma^{\star ,n}$  will be positive when  $\phi$  is a sufficiently powerful activation such as relu or sigmoid and the data points are not repetitive, so the neural network can fit any function of the data. We prove Theorem 3.2 in Section B. Theorem 3.2 can explain why additional over-parametrization has been observed to improve generalization in two-layer networks Neyshabur et al. (2017b). Our margin does not decrease with a larger network size, and therefore Theorem 3.1 gives a better generalization bound. We precisely characterize the value of  $\gamma^{\star ,n}$  in the following section.

# 3.3 THE MAX MARGIN OF NEURAL NETS IS EQUIVALENT TO  $\ell_1$  SVM IN LIFTED SPACE

We link infinite-size neural networks to the  $\ell_1$  SVM over a lifted space, defined via a lifting function  $\varphi : \mathbb{R}^d \to \mathcal{L}^\infty(\mathbb{S}^{d-1})$  mapping data to an infinite feature vector:

$$
x \in \mathbb {R} ^ {d} \rightarrow \varphi (x) \in \mathcal {L} ^ {\infty} \left(\mathbb {S} ^ {d - 1}\right) \text {s a t i s f y i n g} \varphi (x) [ \bar {u} ] = \phi (\bar {u} ^ {\top} x) \tag {3.5}
$$

We look at the margin of linear functionals corresponding to  $\alpha \in \mathcal{L}^1 (\mathbb{S}^{d - 1})$ . The 1-norm SVM over the lifted feature  $\varphi (x)$  solves for the maximum margin:

$$
\gamma_ {\ell_ {1}} \triangleq \max  _ {\alpha} \min  _ {i \in [ n ]} y _ {i} \left\langle \alpha , \varphi \left(x _ {i}\right) \right\rangle \tag {3.6}
$$

$$
\text {s u b j e c t} \| \alpha \| _ {1} \leq 1
$$

where we rely on the inner product and 1-norm defined in Section 1.2. A priori, it is unclear how to optimize this since the kernel trick does not work for  $\ell_1$  norm. Here we will show that optimizing two-layer neural networks with weak regularization is equivalent to solving equation 3.6.

Theorem 3.3. Let  $\gamma_{\ell_1}$  be defined in equation 3.6, and  $\gamma^{\star ,m}$  be defined in Section 3.1. For any  $m\geq n$

$$
\gamma^ {\star , m} = \frac {\gamma_ {\ell_ {1}}}{2} \tag {3.7}
$$

Rosset et al. (2007) and Neyshabur et al. (2014) show a similar equivalence, but between a lifted logistic regression problem and equation 3.1. In contrast, the above theorem, proved in Section B, shows the equivalence between equation 3.1 and the 1-norm SVM when the regularizer is small.

# 3.4 COMPARISON TO KERNEL METHODS

We compare the  $\ell_1$  SVM margin, attainable by a finite neural network, to the  $\ell_2$  margin attainable via kernel methods. Following the setup of Section 3.3, we define the kernel problem over  $\alpha \in \mathcal{L}^2 (\mathbb{S}^{d - 1})$ :

$$
\gamma_ {\ell_ {2}} \triangleq \max  _ {\alpha} \min  _ {i \in [ n ]} y _ {i} \langle \alpha , \varphi (x _ {i}) \rangle \tag {3.8}
$$

subject to  $\sqrt{\kappa}\| \alpha \| _2\leq 1$

where  $\kappa \triangleq \mathrm{Vol}(\mathbb{S}^{d - 1})$ . (We scale  $\| \alpha \| _2$  by  $\sqrt{\kappa}$  to make the lemma statement below cleaner.) First,  $\gamma_{\ell_2}$  can be used to obtain a standard upper bound on the generalization error of the kernel SVM. Following the notation of Section 3.1, we will let  $L_{\ell_2\text{-svm}}$  denote the 0-1 population classification error for the optimizer of equation 3.8.

Lemma 3.4. In the setting of Theorem 3.1, with probability at least  $1 - \delta$ , the generalization error of the standard kernel SVM with relu feature (defined in equation 3.8) is bounded by

![](images/2c29ce671b23844a4c55a654bcc4429d89767b9205cf6e7134ed363a98463852.jpg)  
Figure 1: A visualization of 60 sampled points from  $\mathcal{D}$  in 3 dimensions. Red points denote negative examples and blue points denote positive examples.

$$
L _ {\ell_ {2} - \mathrm {s v m}} \lesssim \frac {1}{\gamma_ {\ell_ {2}}} \sqrt {\frac {D}{d n}} + \epsilon_ {\ell_ {2}} \tag {3.9}
$$

where  $\epsilon_{\ell_2} \triangleq \sqrt{\frac{\log\max\left\{\log_2\frac{C\sqrt{\kappa / d}}{\gamma_{\ell_2}},2\right\}}{n}} + \sqrt{\frac{\log(1 / \delta)}{n}}$  is typically a lower-order term.

The bound above follows from standard techniques (Bartlett & Mendelson, 2002), and we provide a full proof in Section C.1. We construct a data distribution for which this lemma does not give a good bound for kernel methods, but Theorem 3.1 does imply good generalization for two-layer networks.

Theorem 3.5. There exists a data distribution  $p_{\mathrm{data}}$  such that the  $\ell_1$  SVM with relu features has a good margin:

$$
\gamma_ {\ell_ {1}} \gtrsim 1
$$

and with probability  $1 - \delta$  over the choice of i.i.d. samples from  $p_{\mathrm{data}}$ , obtains generalization error

$$
L _ {\ell_ {1} - \operatorname {s v m}} \lesssim \sqrt {\frac {d}{n}} + \epsilon_ {\ell_ {1}}
$$

where  $\epsilon_{\ell_1} \triangleq \sqrt{\frac{\log \log(d \log n)}{n}} + \sqrt{\frac{\log(1 / \delta)}{n}}$  is typically a lower order term. Meanwhile, with high probability the  $\ell_2$  SVM has a small margin:

$$
\gamma_ {\ell_ {2}} \lesssim \max  \left\{\sqrt {\frac {\log n}{n}}, 1 / d \right\}
$$

and therefore the generalization upper bound from Lemma 3.4 is at least

$$
\Omega \left(\min \left\{\frac {1}{\log n}, \frac {d}{\sqrt {n}} \right\}\right)
$$

We briefly overview the construction of  $p_{\mathrm{data}}$  here and defer the full proof of Theorem 3.5 to Section D.1.

Proof sketch for Theorem 3.5. We base  $p_{\mathrm{data}}$  on the distribution  $\mathcal{D}$  of examples  $(x, y)$  described below. Here  $e_i$  is the i-th standard basis vector and we use  $x^\top e_i$  to represent the  $i$ -coordinate of  $x$  (since the subscript is reserved to index training examples).

$$
\left[ \begin{array}{c} e _ {3} ^ {\top} x \\ \vdots \\ e _ {d} ^ {\top} x \end{array} \right] \sim \mathcal {N} (0, I _ {d - 2}), \text {a n d} \left\{ \begin{array}{l l l l} y = + 1, & x ^ {\top} e _ {1} = + 1, & x ^ {\top} e _ {2} = + 1 & \text {w / p r o b .} 1 / 4 \\ y = + 1, & x ^ {\top} e _ {1} = - 1, & x ^ {\top} e _ {2} = - 1 & \text {w / p r o b .} 1 / 4 \\ y = - 1, & x ^ {\top} e _ {1} = + 1, & x ^ {\top} e _ {2} = - 1 & \text {w / p r o b .} 1 / 4 \\ y = - 1, & x ^ {\top} e _ {1} = - 1, & x ^ {\top} e _ {2} = + 1 & \text {w / p r o b .} 1 / 4 \end{array} \right.
$$

Figure 1 shows samples from  $\mathcal{D}$  when there are 3 dimensions. From the visualization, it is clear that there is no linear separator for  $\mathcal{D}$ . As Lemma D.1 shows, a relu network with four neurons can fit this relatively complicated decision boundary. On the other hand, for kernel methods, we prove that the symmetries in  $\mathcal{D}$  induce cancellation in feature space. The following lemmas, proved in Section D.1, formalize this cancellation and show that it results in a small margin for kernel methods.

Lemma 3.6 (Margin upper bound tool). In the setting of Theorem 3.5, we have

$$
\gamma_ {\ell_ {2}} \leq \frac {1}{\sqrt {\kappa}} \cdot \left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \varphi (x _ {i}) y _ {i} \right\| _ {2}
$$

Lemma 3.7. In the setting of Theorem 3.5, let  $(x_{i},y_{i})_{i = 1}^{n}$  be  $n$  i.i.d samples and corresponding labels from  $\mathcal{D}$ . Let  $\varphi$  be defined in equation 3.5 with  $\phi = \mathrm{relu}$ . With high probability (at least  $1 - dn^{-10}$ ), we have

$$
\left\| \frac {1}{n} \sum_ {i = 1} ^ {n} \varphi (x _ {i}) y _ {i} \right\| _ {2} \lesssim \sqrt {\kappa / n} \log n + \sqrt {\kappa} / d
$$

Combining these lemmas gives us the desired bound on  $\gamma_{\ell_2}$ .

Gap in regression setting: We are able to prove an even larger  $\Omega (\sqrt{n / d})$  gap between neural networks and kernel methods in the regression setting where we wish to interpolate continuous labels. Analogously to the classification setting, optimizing a regularized squared error loss on neural networks is equivalent to solving a minimum 1-norm regression problem (see Theorem D.3). Furthermore, kernel methods correspond to a minimum 2-norm problem. We construct distributions  $p_{\mathrm{data}}$  where the 1-norm solution will have a generalization error bound of  $O(\sqrt{d / n})$ , whereas the 2-norm solution will have a generalization error bound that is  $\Omega (1)$  and thus vacuous. In Section D.2, we define the 1-norm and 2-norm regression problems. In Theorem D.6 we formalize our construction.

# 4 PERTURBED WASSERSTEIN GRADIENT FLOW FINDS GLOBAL OPTIMIZERS IN POLYNOMIAL TIME

In the prior section, we studied the limiting behavior of the generalization of a two-layer network as its width goes to infinity. In this section, we will now study the limiting behavior of the optimization algorithm, gradient descent. Prior work (Mei et al., 2018; Chizat & Bach, 2018) has shown that as the hidden layer size grows to infinity, gradient descent for a finite neural network approaches the Wasserstein gradient flow over distributions of hidden units (defined in equation 4.1). Chizat & Bach (2018) also prove that Wasserstein gradient flow converges to a global optimizer in this setting but do not specify a convergence rate.

We show that a perturbed version of Wasserstein gradient flow converges in polynomial time. The informal take-away of this section is that a perturbed version of gradient descent converges in polynomial time on infinite-size neural networks (for the right notion of infinite-size.)

Formally, we optimize the following functional over distributions  $\rho$  on  $\mathbb{R}^{d + 1}$ :

$$
L [ \rho ] \triangleq R \left(\int \Phi d \rho\right) + \int V d \rho
$$

where  $\Phi : \mathbb{R}^{d + 1} \to \mathbb{R}^k$ ,  $R : \mathbb{R}^k \to \mathbb{R}$ , and  $V : \mathbb{R}^{d + 1} \to \mathbb{R}$ . In this work, we consider 2-homogeneous  $\Phi$  and  $V$ . We will additionally require that  $R$  is nonnegative and  $V$  is positive on the unit sphere. Finally, we need standard regularity assumptions on  $R$ ,  $\Phi$ , and  $V$ :

Assumption 4.1 (Regularity conditions on  $\Phi, R, V$ ).  $\Phi$  and  $V$  are differentiable as well as upper bounded and Lipschitz on the unit sphere.  $R$  is Lipschitz and its Hessian has bounded operator norm.

We provide more details on the specific parameters (for boundedness, Lipschitzness, etc.) in Section E.1. We note that relu networks satisfy every condition but differentiability of  $\Phi$ .<sup>5</sup> We can fit a neural network under our framework as follows:

Example 4.2 (Logistic loss for neural networks). We interpret  $\rho$  as a distribution over the parameters of the network. Let  $k \triangleq n$  and  $\Phi_i(\theta) \triangleq w\phi(u^\top x_i)$  for  $\theta = (w, u)$ . In this case,  $\int \Phi d\rho$  is a distributional neural network that computes an output for each of the  $n$  training examples (like a standard neural network, it also computes a weighted sum over hidden units). We can compute the distributional version of the regularized logistic loss in equation 3.1 by setting  $V(\theta) \triangleq \lambda \| \theta \|_2^2$  and  $R(a_1, \ldots, a_n) \triangleq \sum_{i=1}^{n} \log(1 + \exp(-y_i a_i))$ .

We will define  $L^{\prime}[\rho]:\mathbb{R}^{d + 1}\to \mathbb{R}$  with  $L^{\prime}[\rho ](\theta)\triangleq \langle R^{\prime}(\int \Phi d\rho),\Phi (\theta)\rangle +V(\theta)$  and  $v[\rho ](\theta)\triangleq -\nabla_{\theta}L^{\prime}[\rho ](\theta)$ . Informally,  $L^{\prime}[\rho ]$  is the gradient of  $L$  with respect to  $\rho$ , and  $v$  is the induced velocity field. For the standard Wasserstein gradient flow dynamics,  $\rho_{t}$  evolves according to

$$
\frac {d}{d t} \rho_ {t} = - \nabla \cdot (v [ \rho_ {t} ] \rho_ {t}) \tag {4.1}
$$

where  $\nabla$  denotes the divergence of a vector field. For neural networks, these dynamics formally define continuous-time gradient descent when the hidden layer has infinite size (see Theorem 2.6 of Chizat & Bach (2018), for instance).

We propose the following modification of the Wasserstein gradient flow dynamics:

$$
\frac {d}{d t} \rho_ {t} = - \sigma \rho_ {t} + \sigma U ^ {d} - \nabla \cdot (v [ \rho_ {t} ] \rho_ {t}) \tag {4.2}
$$

where  $U^d$  is the uniform distribution on  $\mathbb{S}^d$ . In our perturbed dynamics, we add uniform noise over  $U^d$ . For infinite-size neural networks, one can informally interpret this as re-initializing a very small fraction of the neurons at every step of gradient descent. We prove convergence to a global optimizer in time polynomial in  $1 / \epsilon, d$ , and the regularity parameters.

Theorem 4.3 (Theorem E.4 with regularity parameters omitted). Suppose that  $\Phi$  and  $V$  are 2-homogeneous and the regularity conditions of Assumption 4.1 are satisfied. Also assume that from starting distribution  $\rho_0$ , a solution to the dynamics in equation 4.2 exists. Define  $L^{\star} \triangleq \inf_{\rho} L[\rho]$ . Let  $\epsilon > 0$  be a desired error threshold and choose  $\sigma \triangleq \exp(-d\log(1/\epsilon)\mathrm{poly}(k, L[\rho_0] - L^{\star}))$  and  $t_\epsilon \triangleq \frac{d^2}{\epsilon^4}\mathrm{poly}(\log(1/\epsilon), k, L[\rho_0] - L^{\star})$ , where the regularity parameters for  $\Phi, V$ , and  $R$  are hidden in the  $\mathrm{poly}(\cdot)$ . Then, perturbed Wasserstein gradient flow converges to an  $\epsilon$ -approximate global minimum in  $t_\epsilon$  time:

$$
\min_{0\leq t\leq t_{\epsilon}}L[\rho_{t}] - L^{\star}\leq \epsilon .
$$

We provide a theorem statement that includes regularity parameters in Section E.1. We prove the theorem in Section E.2.

As a technical detail, Theorem 4.3 requires that a solution to the dynamics exists. We can remove this assumption by analyzing a discrete-time version of equation 4.2:

$$
\rho_ {t + 1} \triangleq \rho_ {t} + \eta (- \sigma \rho_ {t} + \sigma U ^ {d} - \nabla \cdot (v [ \rho_ {t} ] \rho_ {t}))
$$

and additionally assuming  $\Phi$  and  $V$  have Lipschitz gradients. In this setting, a polynomial time convergence result also holds. We state the result in Section E.3.

# 5 SIMULATIONS

We first verify the normalized margin convergence on a two-layer networks with one-dimensional input. A single hidden unit computes the following:  $x \mapsto a_j \mathrm{relu}(w_jx + b_j)$ . We add  $\| \cdot \|_2^2$ -regularization to  $a, w$ , and  $b$  and compare the resulting normalized margin to that of an approximate solution of the  $\ell_1$  SVM problem with features  $\mathrm{relu}(wx_i + b)$  for  $w^2 + b^2 = 1$ . Writing this feature vector is intractable, so we solve an approximate version by choosing 1000 evenly spaced values of  $(w, b)$ . Our theory predicts that with decreasing regularization, the margin of the neural network converges to the  $\ell_1$  SVM objective. In Figure 2, we plot this margin convergence and visualize the final networks and ground truth labels. The network margin approaches the ideal one as  $\lambda \to 0$ , and the visualization shows that the network and  $\ell_1$  SVM functions are extremely similar.

Next, we experiment on synthetic data in a higher-dimensional setting. For classification and regression, we compare the generalization error and predicted generalization upper bounds<sup>6</sup> (from

![](images/d9260df709af4b3a9be3d1b0a92c790dc16e1f6d595555d95bcfa96995fe5aa7.jpg)  
Figure 2: Neural network with input dimension 1. Left: Normalized margin as we decrease  $\lambda$ . Right: Visualization of the normalized functions computed by the neural network and  $\ell_1$  SVM solution for  $\lambda \approx 10^{-14}$ .

![](images/881ba015f0816794cfa6aebd9e8715c03654efa074f8977e4298ad631dfe036e.jpg)

![](images/40b4220a1e511b344457b3049f6f84a5aca4dec04076e8f69858a9ff002822b6.jpg)  
Figure 3: Comparing neural networks and kernel methods. Left: Classification. Right: Regression.

![](images/995e31de7d4d3b449327f103fb6462e4236a80369a3c4b3289f919af6eb8016e.jpg)

![](images/8e05be7f830b9128c2e440bb006ff6cb8e8c82b4736d5e104db66cf86e0a82c1.jpg)

![](images/9caf791e465fe3b7aafa42de370f83ea5a389af74805609b2d379c2b83cc3711.jpg)

Theorem 3.1 and Lemmas 3.4, D.4, and D.5) of a trained neural network against a  $\ell_2$  kernel SVM with relu features as we vary  $n$ . For classification we plot 0-1 error, whereas for regression we plot squared error. Our ground truth comes from a random neural network with 6 hidden units. For classification, we used rejection sampling to obtain datapoints with unnormalized margin of at least 0.1 on the ground truth network. We use a fixed dimension of  $d = 20$ . For all experiments, we train the network for 20000 steps with  $\lambda = 10^{-8}$  and average over 100 trials for each plot point.

The plots in Figure 3 show that two-layer networks clearly outperform kernel methods in test error as  $n$  grows. However, there seems to be looseness in the upper bounds for kernel methods: the kernel generalization bound appears to stay constant with  $n$  (as predicted by our theory for regression), but the kernel test error decreases. There is also some variance in the neural network generalization bound for classification. This occurred likely because we did not tune learning rate and training time, so the optimization failed to find the best margin.

In Section F, we include additional experiments training modified WideResNet architectures on CIFAR10 and CIFAR100. Although ResNet is not homogeneous, we still report interesting increases in generalization performance from annealing the weight decay during training, versus staying at a fixed decay rate.

# 6 CONCLUSION

We have made the case that maximizing margin is one of the inductive biases of relu networks with cross-entropy loss. We show that we can obtain a maximum normalized margin by training with a weak regularizer. We also prove that larger  $\ell_2$ -normalized margin indicates better generalization for two-layer nets. Our work leaves open the question of how the  $\ell_2$ -normalized margin relates to generalization in much deeper neural networks. This is a fascinating theoretical and empirical question for future work. On the optimization side, we make progress towards understanding overparametrized gradient descent by analyzing infinite-size neural networks. A natural direction for future work is to apply our theory to optimize the margin of finite-sized neural networks.

# REFERENCES

Sanjeev Arora, Nadav Cohen, and Elad Hazan. On the optimization of deep networks: Implicit acceleration by overparameterization. arXiv preprint arXiv:1802.06509, 2018a.  
Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. arXiv preprint arXiv:1802.05296, 2018b.  
Keith Ball et al. An elementary introduction to modern convex geometry. Flavors of geometry, 31: 1-58, 1997.  
Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3(Nov):463-482, 2002.  
Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. In Advances in Neural Information Processing Systems, pp. 6240-6249, 2017.  
Mikhail Belkin, Siyuan Ma, and Soumik Mandal. To understand deep learning we need to understand kernel learning. arXiv preprint arXiv:1802.01396, 2018.  
Alon Brutzkus, Amir Globerson, Eran Malach, and Shai Shalev-Shwartz. Sgd learns overparameterized networks that provably generalize on linearly separable data. arXiv preprint arXiv:1710.10174, 2017.  
Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. Entropy-sgd: Biasing gradient descent into wide valleys. arXiv preprint arXiv:1611.01838, 2016.  
Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for over-parameterized models using optimal transport. arXiv preprint arXiv:1805.09545, 2018.  
Simon S Du and Jason D Lee. On the power of over-parametrization in neural networks with quadratic activation. arXiv preprint arXiv:1803.01206, 2018.  
Simon S Du, Jason D Lee, Yuandong Tian, Barnabas Poczos, and Aarti Singh. Gradient descent learns one-hidden-layer cnn: Don't be afraid of spurious local minima. arXiv preprint arXiv:1712.00779, 2017.  
Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. arXiv preprint arXiv:1703.11008, 2017.  
Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. arXiv preprint arXiv:1712.06541, 2017.  
Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit regularization in matrix factorization. In Advances in Neural Information Processing Systems, pp. 6151-6159, 2017.  
Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias in terms of optimization geometry. arXiv preprint arXiv:1802.08246, 2018a.  
Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Implicit bias of gradient descent on linear convolutional networks. arXiv preprint arXiv:1806.00468, 2018b.  
Benjamin D Haeffele and René Vidal. Global optimality in tensor factorization, deep learning, and beyond. arXiv preprint arXiv:1506.07540, 2015.  
Moritz Hardt, Benjamin Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. arXiv preprint arXiv:1509.01240, 2015.  
Ziwei Ji and Matus Telgarsky. Risk and parameter convergence of logistic regression. arXiv preprint arXiv:1803.07300, 2018.

Sham M Kakade, Karthik Sridharan, and Ambuj Tewari. On the complexity of linear prediction: Risk bounds, margin bounds, and regularization. In Advances in neural information processing systems, pp. 793-800, 2009.  
Vladimir Koltchinskii, Dmitry Panchenko, et al. Empirical margin distributions and bounding the generalization error of combined classifiers. The Annals of Statistics, 30(1):1-50, 2002.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. Algorithmic regularization in over-parameterized matrix sensing and neural networks with quadratic activations. In Conference On Learning Theory, pp. 2-47, 2018.  
T. Liang and A. Rakhlin. Just Interpolate: Kernel "Ridgeless" Regression Can Generalize. ArXiv e-prints, August 2018.  
Roi Livni, Shai Shalev-Shwartz, and Ohad Shamir. On the computational efficiency of training neural networks. In Advances in Neural Information Processing Systems, pp. 855-863, 2014.  
Cong Ma, Kaizheng Wang, Yuejie Chi, and Yuxin Chen. Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval, matrix completion and blind deconvolution. arXiv preprint arXiv:1711.10467, 2017.  
Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layers neural networks. arXiv preprint arXiv:1804.06561, 2018.  
Ari S Morcos, David GT Barrett, Neil C Rabinowitz, and Matthew Botvinick. On the importance of single directions for generalization. arXiv preprint arXiv:1803.06959, 2018.  
Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. In search of the real inductive bias: On the role of implicit regularization in deep learning. arXiv preprint arXiv:1412.6614, 2014.  
Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In Advances in Neural Information Processing Systems, pp. 2422-2430, 2015a.  
Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Conference on Learning Theory, pp. 1376-1401, 2015b.  
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nathan Srebro. A pac-bayesian approach to spectrally-normalized margin bounds for neural networks. arXiv preprint arXiv:1707.09564, 2017a.  
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in deep learning. In Advances in Neural Information Processing Systems, pp. 5947-5956, 2017b.  
Behnam Neyshabur, Zhiyuan Li, Srinadh Bhojanapalli, Yann LeCun, and Nathan Srebro. Towards understanding the role of over-parametrization in generalization of neural networks. arXiv preprint arXiv:1805.12076, 2018.  
Quynh Nguyen and Matthias Hein. The loss surface of deep and wide neural networks. arXiv preprint arXiv:1704.08045, 2017.  
Saharon Rosset, Ji Zhu, and Trevor Hastie. Boosting as a regularized path to a maximum margin classifier. Journal of Machine Learning Research, 5(Aug):941-973, 2004.  
Saharon Rosset, Grzegorz Swirszcz, Nathan Srebro, and Ji Zhu. 11 regularization in infinite dimensional feature spaces. In International Conference on Computational Learning Theory, pp. 544-558. Springer, 2007.  
Itay Safran and Ohad Shamir. On the quality of the initial basin in overspecified neural networks. In International Conference on Machine Learning, pp. 774-782, 2016.

Levent Sagun, Utku Evci, V Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017.  
Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of neural networks. arXiv preprint arXiv:1805.01053, 2018.  
Daniel Soudry and Yair Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. arXiv preprint arXiv:1605.08361, 2016.  
Daniel Soudry, Elad Hoffer, and Nathan Srebro. The implicit bias of gradient descent on separable data. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=r1q7n9gAb.  
Ryan J Tibshirani et al. The lasso problem and uniqueness. Electronic Journal of Statistics, 7: 1456-1490, 2013.  
Twan van Laarhoven. L2 regularization versus batch and weight normalization. arXiv preprint arXiv:1706.05350, 2017.  
Luca Venturi, Afonso Bandeira, and Joan Bruna. Neural networks with finite intrinsic dimension have no spurious valleys. arXiv preprint arXiv:1802.06384, 2018.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.  
Ji Zhu, Saharon Rosset, Robert Tibshirani, and Trevor J Hastie. 1-norm support vector machines. In Advances in neural information processing systems, pp. 49-56, 2004.
