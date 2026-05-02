# UNDERSTANDING STRAIGHT-THROUGH ESTIMATOR IN TRAINING ACTIVATION QUANTIZED NEURAL NETS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Training activation quantized neural networks involves piecewise constant loss functions with the sampled gradient vanishing almost everywhere, which is undesirable for back-propagation. An empirical way around this issue is to use a straight-through estimator (STE) (Bengio et al., 2013) in the backward pass, so that the resulting unusual "gradient" becomes non-trivial. In this paper, we make the first theoretical justification for the concept of STE, by considering the problem of learning a one-hidden-layer convolutional network with binarized ReLU activation and Gaussian input data. We refer to the unusual "gradient" based on STE as coarse gradient, which essentially is not the gradient of any function. Apparently, the choice of STE is not unique. We prove that if the STE is properly chosen, the negative expected coarse gradient is a descent direction for minimizing the population loss, and the associated coarse gradient descent algorithm converges to a local minimum (more rigorously, a critical point) of the population loss minimization problem. Moreover, we show that a relatively poor choice of STE may lead to instability of the training algorithm near certain local minima, which is also validated by our CIFAR-10 experiments.

# 1 INTRODUCTION

Deep neural networks (DNN) have achieved the remarkable success in many machine learning applications such as computer vision (Krizhevsky et al., 2012; Ren et al., 2015), natural language processing (Collobert & Weston, 2008) and reinforcement learning (Mnih et al., 2015; Silver et al., 2016). However, the deployment of DNN typically require hundreds of megabytes of memory storage for the trainable full-precision floating-point parameters, and billions of floating-point operations to make a single inference. To achieve the compression and acceleration, many recent efforts have been made to the training of quantized DNN, in the hope of maintaining the performance of their float counterparts (Courbariaux et al., 2015; Rastegari et al., 2016; Cai et al., 2017).

Training fully quantized DNN amounts to solving a very challenging optimization problem. It calls for minimizing a piecewise constant and highly nonconvex empirical risk function  $f(\boldsymbol{w})$  subject to a discrete set-constraint  $\boldsymbol{w} \in \mathcal{Q}$  that characterizes the quantized weights. In particular, weight quantization of DNN have been extensively studied in the literature; see for examples (Li et al., 2016; Zhu et al., 2016; Zhou et al., 2017; Li et al., 2017; Hou & Kwok, 2018). On the other hand, the gradient  $\nabla f(\boldsymbol{w})$  in training activation quantized DNN is almost everywhere (a.e.) zero, which makes the standard back-propagation inapplicable. The arguably most effective way around this issue is nothing but to construct a non-trivial search direction by properly modifying the chain rule. Specifically, one can replace the a.e. zero derivative of quantized activation function composited in the chain rule with a related surrogate. This proxy derivative used in the backward pass only is referred as the straight-through estimator (STE) (Bengio et al., 2013).

# 1.1 RELATED WORKS

The concept of STE was originally introduced in lecture 9c of (Hinton, 2012) for training networks with the hard threshold activation  $1_{\{x > 0\}}$  (a.k.a. binary neuron). (Hinton, 2012) proposed to simply back-propagate through the hard threshold function as if it had been the identity function. (Bengio et al., 2013) proposed a STE variant which uses the derivative of the sigmoid function instead. In the

training of DNN with weights and activations constrained to  $\pm 1$ , (Hubara et al., 2016) substituted the derivative of the signum activation function with  $1_{\{|x|\leq 1\}}$  in the backward pass. Later the idea of STE was readily extended to the training of DNN with general quantized ReLU activations (Hubara et al., 2018; Zhou et al., 2016; Cai et al., 2017; Choi et al., 2018), where some other proxies took place including the derivatives of vanilla ReLU and clipped ReLU. Despite all the empirical success of STE, to our best knowledge, there is almost no theoretical understanding of why it works.

Similar scenarios, where the derivative of certain layer composited in the loss function is not desirable for back-propagation, have also been brought up recently by (Wang et al., 2018) and (Athalye et al., 2018). The former proposed an implicit weighted nonlocal Laplacian layer as the classifier to improve the generalization accuracy of DNN. In the backward pass, the derivative of a pre-trained fully-connected layer was used as a surrogate. To circumvent the defense in adversarial attack (Szegedy et al., 2013), (Athalye et al., 2018) introduced the so-called backward pass differentiable approximation to deal with the obfuscated gradients, which shares the same spirit as STE. Again, neither of these two papers theoretically justified the proposed training approach.

Another line of research studies the convergence of (stochastic) gradient descent algorithm for learning shallow ReLU nets with one or two linear layers and Gaussian input data. Some works consider the empirical risk minimization with finite input samples (Zhong et al., 2017; Soltanolkotabi, 2017), while some others consider the minimization of population loss averaged over the whole data space (Brutzkus & Globerson, 2017; Tian, 2017; Li & Yuan, 2017; Du et al., 2018). The advantage of using the population loss model is that it admits analytic formulas for both the objective function and gradient, which facilitates the analysis. It is of the common interest to analyze whether (or under what conditions) the (stochastic) gradient descent with the standard back-propagation converges to the global minimum of the regression problem and thus recovers the true weights.

# 1.2 MAIN CONTRIBUTIONS

Throughout this paper, we shall refer to the resultant composite "gradient" through STE as coarse gradient. The coarse gradient is basically not the gradient of any function. Our key contribution is to provide the first theoretical understanding of STE by analyzing the coarse gradient descent algorithm for learning a one-hidden-layer network with binary activation and Gaussian data. We consider two representative STEs: the derivatives of the identity function (Hinton, 2012) and the vanilla ReLU (Cai et al., 2017), and adopt the model of population loss minimization. We derive the explicit form of the expected coarse gradients corresponding to the two STEs, and show that the negative expected coarse gradient based on vanilla ReLU is a descent direction for the minimizing the population loss, whereas this is not necessarily true for the one based on the identity function. Moreover, we prove that the former guarantees the convergence to a critical point (a saddle point or a (local) minimizer) and the latter can be unstable sometimes near certain local minima. Indeed, in our experiment on CIFAR-10 using ResNet-20, it is observed that the training algorithm using the identity STE is repelled from good minima and converges to an inferior one with higher training loss and decreased generalization accuracy. This is an implication of the poor performance of the identity STE, not because of the slow convergence of its corresponding coarse gradient descent, instead due to the fact that the algorithm can never reach a good local minimum.

Notations.  $\| \cdot \|$  denotes the Euclidean norm of a vector or the spectral norm of a matrix.  $\mathbf{0}_n\in \mathbb{R}^n$  represents the vector of all zeros, whereas  $\mathbf{1}_n\in \mathbb{R}^n$  the vector of all ones.  $I_{n}$  is the identity matrix of order  $n$ . For any  $\boldsymbol {w}$ ,  $\mathbf{z}\in \mathbb{R}^n$ ,  $\boldsymbol{w}^{\top}\mathbf{z} = \langle \boldsymbol {w},\mathbf{z}\rangle = \sum_{i}w_{i}z_{i}$  is their inner product.  $\boldsymbol {w}\odot \mathbf{z}$  denotes the Hadamard product whose  $i$ th entry is given by  $(\boldsymbol {w}\odot \mathbf{z})_i = w_i z_i$ .

# 2 LEARNING ONE-HIDDEN-LAYER CNN WITH BINARY ACTIVATION

In this paper, we consider a one-hidden-layer network model (Du et al., 2018) that outputs the prediction

$$
y (\mathbf {Z}, \boldsymbol {v}, \boldsymbol {w}) := \sum_ {i = 1} ^ {m} v _ {i} \sigma \left(\mathbf {Z} _ {i} ^ {\top} \boldsymbol {w}\right) = \boldsymbol {v} ^ {\top} \sigma \left(\mathbf {Z} \boldsymbol {w}\right)
$$

for some input  $\mathbf{Z} \in \mathbb{R}^{m \times n}$ . Here  $\boldsymbol{w} \in \mathbb{R}^n$  and  $\boldsymbol{v} \in \mathbb{R}^m$  are the trainable weights in the first and second linear layer, respectively;  $\mathbf{Z}_i^\top$  denotes the  $i$ th row vector of  $\mathbf{Z}$ ; the activation function  $\sigma$  acts

component-wise on the vector  $\mathbf{Z}\mathbf{w}$ , i.e.,  $\sigma (\mathbf{Z}\mathbf{w})_i = \sigma ((\mathbf{Z}\mathbf{w})_i) = \sigma (\mathbf{Z}_i^\top \mathbf{w})$ . The first layer serves as a convolutional layer, where each row  $\mathbf{Z}_i^\top$  can be viewed as a patch sampled from  $\mathbf{Z}$  and the weight filter  $\mathbf{w}$  is shared among all patches, and the second linear layer is the classifier. The label is generated according to  $y^{*}(\mathbf{Z}) = (v^{*})^{\top}\sigma (\mathbf{Z}w^{*})$  for some true (non-zero) parameters  $v^{*}$  and  $w^{*}$ . Moreover, we use the following squared sample loss

$$
\ell (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) := \frac {1}{2} (y (\mathbf {Z}, \boldsymbol {v}, \boldsymbol {w}) - y ^ {*} (\mathbf {Z})) ^ {2} = \frac {1}{2} \left(\boldsymbol {v} ^ {\top} \sigma (\mathbf {Z} \boldsymbol {w}) - y ^ {*} (\mathbf {Z})\right) ^ {2}. \tag {1}
$$

Unlike in (Du et al., 2018), the activation function  $\sigma$  here is not ReLU, but the binary function  $\sigma(x) = 1_{\{x > 0\}}$ , same as the hard threshold activation (Hinton, 2012).

We assume that the entries of  $\mathbf{Z} \in \mathbb{R}^{m \times n}$  are i.i.d. sampled from the Gaussian distribution  $\mathcal{N}(0,1)$  (Zhong et al., 2017; Brutzkus & Globerson, 2017). The legitimacy of this assumption comes from the use of batch normalization (Ioffe & Szegedy, 2015) in most architectures, which sends normalized inputs to the linear layers. Since  $\ell(\boldsymbol{v}, \boldsymbol{w}; \mathbf{Z}) = \ell(\boldsymbol{v}, \boldsymbol{w}/c; \mathbf{Z})$  for any scalar  $c > 0$ , without loss of generality, we take  $\| \boldsymbol{w}^* \| = 1$  and cast the learning task as the following population loss minimization problem:

$$
\min  _ {\boldsymbol {v} \in \mathbb {R} ^ {m}, \boldsymbol {w} \in \mathbb {R} ^ {n}} f (\boldsymbol {v}, \boldsymbol {w}) := \mathbb {E} _ {\boldsymbol {Z}} [ \ell (\boldsymbol {v}, \boldsymbol {w}; \boldsymbol {Z}) ], \tag {2}
$$

where the sample loss  $\ell (\pmb {v},\pmb {w};\mathbf{Z})$  is given by (1).

# 2.1 BACK-PROPAGATION AND COARSE GRADIENT DESCENT

With the Gaussian assumption on  $\mathbf{Z}$ , as will be shown in section 2.2, it is possible to find the analytic expressions of  $f(\boldsymbol{v},\boldsymbol{w})$  and its gradient

$$
\nabla f (\boldsymbol {v}, \boldsymbol {w}) := \left[ \begin{array}{c} \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}) \\ \frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}) \end{array} \right].
$$

The information about  $\nabla f(\pmb{v},\pmb{w})$ , however, is not available for the network training. In fact, we can only use the expectation of the sample gradient, namely,

$$
\mathbb {E} _ {\mathbf {Z}} \left[ \frac {\partial \ell}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right] \text {a n d} \mathbb {E} _ {\mathbf {Z}} \left[ \frac {\partial \ell}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right].
$$

We remark that the expected partial gradient  $\mathbb{E}_{\mathbf{Z}}\left[\frac{\partial\ell}{\partial\boldsymbol{w}} (\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  is not the same as  $\frac{\partial f}{\partial\boldsymbol{w}} (\boldsymbol {v},\boldsymbol {w}) = \frac{\partial\mathbb{E}_{\mathbf{Z}}[\ell(\boldsymbol{v},\boldsymbol{w};\mathbf{Z})]}{\partial\boldsymbol{w}}$  . By the standard back-propagation or chain rule, we readily check that

$$
\frac {\partial \ell}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) = \sigma (\mathbf {Z} \boldsymbol {w}) \left(\boldsymbol {v} ^ {\top} \sigma (\mathbf {Z} \boldsymbol {w}) - y ^ {*} (\mathbf {Z})\right) \tag {3}
$$

and

$$
\frac {\partial \ell}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) = \mathbf {Z} ^ {\top} \left(\sigma^ {\prime} (\mathbf {Z} \boldsymbol {w}) \odot \boldsymbol {v}\right) \left(\boldsymbol {v} ^ {\top} \sigma (\mathbf {Z} \boldsymbol {w}) - y ^ {*} (\mathbf {Z})\right). \tag {4}
$$

Note that  $\sigma'$  is zero a.e., which makes (4) inapplicable to the training. The idea of STE is to simply replace the a.e. zero component  $\sigma'$  in (4) with a related non-trivial function  $\mu'$  (Hinton, 2012; Bengio et al., 2013; Hubara et al., 2016; Cai et al., 2017), which is the derivative of some (sub)differentiable function  $\mu$ . More precisely, back-propagation using the STE  $\mu'$  gives the following non-trivial surrogate of  $\frac{\partial\ell}{\partial\boldsymbol{w}}(\boldsymbol{v},\boldsymbol{w};\mathbf{Z})$ , to which we refer as the coarse (partial) gradient

$$
\boldsymbol {g} _ {\mu} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) = \mathbf {Z} ^ {\top} \left(\mu^ {\prime} (\mathbf {Z} \boldsymbol {w}) \odot \boldsymbol {v}\right) \left(\boldsymbol {v} ^ {\top} \sigma (\mathbf {Z} \boldsymbol {w}) - y ^ {*} (\mathbf {Z})\right). \tag {5}
$$

Using the STE  $\mu^{\prime}$  to train the one-hidden-layer convolutional neural network (CNN) with binary activation gives rise to the (full-batch) coarse gradient descent described in Algorithm 1.

# 2.2 PRELIMINARIES

Let us present some preliminaries about the landscape of the population loss function  $f(\boldsymbol{v}, \boldsymbol{w})$ . To this end, we define the angle between  $\boldsymbol{w}$  and  $\boldsymbol{w}^*$  as  $\theta(\boldsymbol{w}, \boldsymbol{w}^*) := \arccos \left( \frac{\boldsymbol{w}^\top \boldsymbol{w}^*}{\|\boldsymbol{w}\| \|\boldsymbol{w}^*\|} \right)$  for any  $\boldsymbol{w} \neq \mathbf{0}_n$ . Recall that the label is given by  $y^*(\mathbf{Z}) = (\boldsymbol{v}^*)^\top \mathbf{Z} \boldsymbol{w}^*$ , we elaborate on the expressions of  $f(\boldsymbol{v}, \boldsymbol{w})$  and  $\nabla f(\boldsymbol{v}, \boldsymbol{w})$ .

Algorithm 1 Coarse gradient descent for learning one-hidden-layer CNN with STE  $\mu^{\prime}$

Input: initialization  $\pmb{v}^0\in \mathbb{R}^m$ $\pmb{w}^{0}\in \mathbb{R}^{n}$  , learning rate  $\eta$

for  $t = 1,2,\ldots$  do

$$
\boldsymbol {v} ^ {t + 1} = \boldsymbol {v} ^ {t} - \eta \mathbb {E} _ {\boldsymbol {Z}} \left[ \frac {\partial \boldsymbol {\ell}}{\partial \boldsymbol {v}} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}; \boldsymbol {Z}) \right]
$$

$$
\boldsymbol {w} ^ {t + 1} = \boldsymbol {w} ^ {t} - \eta \mathbb {E} _ {\mathbf {Z}} \left[ \bar {\boldsymbol {g}} _ {\mu} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}; \mathbf {Z}) \right]
$$

end for

Lemma 1. Suppose all entries of  $\mathbf{Z}$  follow i.i.d.  $\mathcal{N}(0,1)$ . If  $\boldsymbol{w} \neq \mathbf{0}_n$ , the population loss  $f(\boldsymbol{v}, \boldsymbol{w})$  is given by

$$
\frac {1}{8} \left[ \boldsymbol {v} ^ {\top} \left(\boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) \boldsymbol {v} - 2 \boldsymbol {v} ^ {\top} \left(\left(1 - \frac {2}{\pi} \theta (\boldsymbol {w}, \boldsymbol {w} ^ {*})\right) \boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) \boldsymbol {v} ^ {*} + (\boldsymbol {v} ^ {*}) ^ {\top} \left(\boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) \boldsymbol {v} ^ {*} \right].
$$

In addition,  $f(\pmb{v}, \pmb{w}) = \frac{1}{8} (\pmb{v}^*)^\top (\pmb{I}_m + \pmb{1}_m\pmb{1}_m^\top) \pmb{v}^*$  for  $\pmb{w} = \mathbf{0}_n$ .

All the technical proofs in this paper will be detailed in the appendix.

Lemma 2. If  $\pmb{w} \neq \mathbf{0}_n$  and  $\theta(\pmb{w}, \pmb{w}^*) \in (0, \pi)$ , the partial gradients of  $f(\pmb{v}, \pmb{w})$  w.r.t.  $\pmb{v}$  and  $\pmb{w}$  are

$$
\frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}) = \frac {1}{4} \left(\boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) \boldsymbol {v} - \frac {1}{4} \left(\left(1 - \frac {2}{\pi} \theta (\boldsymbol {w}, \boldsymbol {w} ^ {*})\right) \boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) \boldsymbol {v} ^ {*} \tag {6}
$$

and

$$
\frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}) = - \frac {\boldsymbol {v} ^ {\top} \boldsymbol {v} ^ {*}}{2 \pi \| \boldsymbol {w} \|} \frac {\left(\boldsymbol {I} _ {n} - \frac {\boldsymbol {w} \boldsymbol {w} ^ {\top}}{\| \boldsymbol {w} \| ^ {2}}\right) \boldsymbol {w} ^ {*}}{\left\| \left(\boldsymbol {I} _ {n} - \frac {\boldsymbol {w} \boldsymbol {w} ^ {\top}}{\| \boldsymbol {w} \| ^ {2}}\right) \boldsymbol {w} ^ {*} \right\|}, \tag {7}
$$

respectively.

For any  $\pmb{v} \in \mathbb{R}^m$ ,  $(\pmb{v}, \mathbf{0}_m)$  is impossible to be a local minimizer. The only possible (local) minimizers of the model (2) are located at

1. Stationary points where the gradients given by (6) and (7) vanish simultaneously (which may not be possible), i.e.,

$$
\boldsymbol {v} ^ {\top} \boldsymbol {v} ^ {*} = 0 \text {a n d} \boldsymbol {v} = \left(\boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) ^ {- 1} \left(\left(1 - \frac {2}{\pi} \theta (\boldsymbol {w}, \boldsymbol {w} ^ {*})\right) \boldsymbol {I} _ {m} + \boldsymbol {1} _ {m} \boldsymbol {1} _ {m} ^ {\top}\right) \boldsymbol {v} ^ {*}. \tag {8}
$$

2. Non-differentiable points where  $\theta(\boldsymbol{w}, \boldsymbol{w}^*) = 0$  and  $\boldsymbol{v} = \boldsymbol{v}^*$ , or  $\theta(\boldsymbol{w}, \boldsymbol{w}^*) = \pi$  and  $\boldsymbol{v} = (I_m + \mathbf{1}_m \mathbf{1}_m^\top)^{-1} (\mathbf{1}_m \mathbf{1}_m^\top - \mathbf{I}_m) \boldsymbol{v}^*$ .

Among them,  $\{(\pmb{v},\pmb{w}):\pmb{v} = \pmb{v}^{*},\theta (\pmb{w},\pmb{w}^{*}) = 0\}$  are obviously the global minimizers of (2). We show that the stationary points, if exist, can only be saddle points, and  $\{(v,w):\theta (w,w^{*}) = \pi ,\pmb {v} = \left(\pmb {I}_m + \pmb {1}_m\pmb {1}_m^\top\right)^{-1}(\pmb {1}_m\pmb {1}_m^\top -\pmb {I}_m)\pmb {v}^*\}$  are the only potential spurious local minimizers.

Proposition 1. If the true parameter  $\pmb{v}^{*}$  satisfies  $(\mathbf{1}_m^\top \pmb{v}^*)^2 < \frac{m + 1}{2}\| \pmb{v}^*\|^2$ , then

$$
\left\{\left(\boldsymbol {v}, \boldsymbol {w}\right): \boldsymbol {v} = \left(\boldsymbol {I} _ {m} + \mathbf {1} _ {m} \mathbf {1} _ {m} ^ {\top}\right) ^ {- 1} \left(\frac {- \left(\mathbf {1} _ {m} ^ {\top} \boldsymbol {v} ^ {*}\right) ^ {2}}{(m + 1) \| \boldsymbol {v} ^ {*} \| ^ {2} - \left(\mathbf {1} _ {m} ^ {\top} \boldsymbol {v} ^ {*}\right) ^ {2}} \boldsymbol {I} _ {m} + \mathbf {1} _ {m} \mathbf {1} _ {m} ^ {\top}\right) \boldsymbol {v} ^ {*}, \right.
$$

$$
\left. \theta (\boldsymbol {w}, \boldsymbol {w} ^ {*}) = \frac {\pi}{2} \frac {(m + 1) \| \boldsymbol {v} ^ {*} \| ^ {2}}{(m + 1) \| \boldsymbol {v} ^ {*} \| ^ {2} - \left(\mathbf {1} _ {m} ^ {\top} \boldsymbol {v} ^ {*}\right) ^ {2}} \right\} \tag {9}
$$

give the saddle points obeying (8), and  $\{(\pmb {v},\pmb {w}):\theta (\pmb {w},\pmb{w}^{*}) = \pi ,\pmb {v} = \left(\pmb{I}_{m} + \pmb{1}_{m}\pmb{1}_{m}^{\top}\right)^{-1}(\pmb{1}_{m}\pmb{1}_{m}^{\top} - \pmb{I}_{m})\pmb{v}^{*}\}$  are the spurious local minimizers. Otherwise, the model (2) has no saddle points or spurious local minimizers.

We further prove that the underlying true gradient  $\nabla f(\pmb{v},\pmb{w})$  given by (6) and (7), is Lipschitz continuous under a boundedness condition.

Lemma 3. For any differentiable points  $(\pmb{v},\pmb{w})$  and  $(\tilde{\pmb{v}},\tilde{\pmb{w}})$  with  $\min \{\| \pmb {w}\| ,\| \tilde{\pmb{w}}\| \} = c_{\pmb{w}} > 0$  and  $\max \{\| \pmb {v}\| ,\| \tilde{\pmb{v}}\| \} = C_v$  there exists a Lipschitz constant  $L > 0$  depending on  $C_\nu$  and  $c_{\pmb{w}}$  such that

$$
\| \nabla f (\boldsymbol {v}, \boldsymbol {w}) - \nabla f (\tilde {\boldsymbol {v}}, \tilde {\boldsymbol {w}}) \| \leq L \| (\boldsymbol {v}, \boldsymbol {w}) - (\tilde {\boldsymbol {v}}, \tilde {\boldsymbol {w}}) \|.
$$

# 3 MAIN RESULTS

We are most interested in the complex case where both the saddle points and spurious local minimizers are present. Our main results are concerned with the behaviors of the coarse gradient descent summarized in Algorithm 1 when the derivatives of the vanilla ReLU and identity function serve as the STE, respectively. Intuitively, the ReLU STE is supposed to outperform the identity STE, because ReLU is obviously a better approximation to the activation function  $\sigma(x) = 1_{\{x > 0\}}$ .

Theorem 1. Let  $\{(\pmb{v}^t, \pmb{w}^t)\}$  be the sequence generated by Algorithm 1 with  $ReLU\mu(x) = \max\{x, 0\}$ . Suppose  $\| \pmb{v}^t \| \leq C_\pmb{v}$  and  $\| \pmb{w}^t \| \geq c_\pmb{w}$  for all  $t$  with some  $C_\pmb{v}, c_\pmb{w} > 0$ . Then if the learning rate  $\eta > 0$  is sufficiently small, for any initialization  $(\pmb{v}^0, \pmb{w}^0)$ , the objective sequence  $\{f(\pmb{v}^t, \pmb{w}^t)\}$  is monotonically decreasing, and  $\{(v^t, w^t)\}$  converges to a saddle point or a (local) minimizer of the population loss minimization (2). In addition, if  $\mathbf{1}_m^\top \pmb{v}^* \neq 0$  and  $m > 1$ , the descent and convergence properties do not hold for Algorithm 1 with the identity function  $\mu(x) = x$  near the local minimizers satisfying  $\theta(\pmb{w}, \pmb{w}^*) = \pi$  and  $\pmb{v} = (\pmb{I}_m + \pmb{1}_m \pmb{1}_m^\top)^{-1} (\pmb{1}_m \pmb{1}_m^\top - \pmb{I}_m) \pmb{v}^*$ .

Remark 1. The convergence guarantee for the coarse gradient descent using vanilla ReLU is established under the assumption that there are infinite training samples. When there are only a few data, in a coarse scale, the empirical loss is roughly descending along the direction of negative coarse gradient, as illustrated by Figure 1. As the sample size increases, the empirical loss gains monotonicity and smoothness. This is different from the conventional gradient descent widely studied in the existing literature, which enjoys the descent property regardless of the sample size.

In the rest of this section, we sketch the mathematical analysis for the main results.

![](images/506acf6d368c3ab73c85f153a299001d468cbbfcbec66e55cb2ca1da1f7f27fe.jpg)  
Figure 1: The plots of the empirical loss moving by one step in the direction of negative coarse gradient v.s. the learning rate (step size)  $\eta$  for different sample sizes.

![](images/2a0d80b8aad1fcc9cb2d51142f9113810df39a6fe637f891088e770618fb57cb.jpg)

![](images/561abc20c4b6c4090c50fb5cc8e7ed76552fa5837b57da5afb5831eae803198d.jpg)

# 3.1 DERIVATIVE OF THE VANILLA RELU AS STE

If we choose the derivative of  $\operatorname{ReLU} \mu(x) = \max\{x, 0\}$  as the STE in (5), it is easy to see  $\mu'(x) = \sigma(x)$ , and we have the following expressions of  $\mathbb{E}_{\mathbf{Z}}\left[\frac{\partial \ell}{\partial \boldsymbol{v}}(\boldsymbol{v}, \boldsymbol{w}; \mathbf{Z})\right]$  and  $\mathbb{E}_{\mathbf{Z}}\left[g_{\mathrm{relu}}(\boldsymbol{v}, \boldsymbol{w}; \mathbf{Z})\right]$  for Algorithm 1.

Lemma 4. The expected partial gradient of  $\ell (\pmb {v},\pmb {w};\mathbf{Z})$  w.r.t.  $\pmb{v}$  is

$$
\mathbb {E} _ {\mathbf {Z}} \left[ \frac {\partial \ell}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right] = \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}). \tag {10}
$$

Let  $\mu (x) = \max \{x,0\}$  in (5). The expected coarse gradient w.r.t.  $\pmb{w}$  is

$$
\mathbb {E} _ {\mathbf {Z}} \left[ g _ {\text {r e l u}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right] = \frac {h (\boldsymbol {v} , \boldsymbol {v} ^ {*})}{2 \sqrt {2 \pi}} \frac {\boldsymbol {w}}{\| \boldsymbol {w} \|} - \cos \left(\frac {\theta (\boldsymbol {w} , \boldsymbol {w} ^ {*})}{2}\right) \frac {\boldsymbol {v} ^ {\top} \boldsymbol {v} ^ {*}}{\sqrt {2 \pi}} \frac {\frac {\boldsymbol {w}}{\| \boldsymbol {w} \|} + \boldsymbol {w} ^ {*}}{\left\| \frac {\boldsymbol {w}}{\| \boldsymbol {w} \|} + \boldsymbol {w} ^ {*} \right\|}, ^ {1} \tag {11}
$$

where  $h(\pmb{v},\pmb{v}^{*}) = \| \pmb{v}\|^{2} + (\mathbf{1}_{m}^{\top}\pmb{v})^{2} - (\mathbf{1}_{m}^{\top}\pmb{v})(\mathbf{1}_{m}^{\top}\pmb{v}^{*}) + \pmb{v}^{\top}\pmb{v}^{*}$

The key observation is that the coarse partial gradient  $\mathbb{E}_{\mathbf{Z}}\left[g_{\mathrm{relu}}(\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  has non-negative correlation with the true gradient  $\frac{\partial f}{\partial w} (\boldsymbol {v},\boldsymbol {w})$  , and  $-\mathbb{E}_{\mathbf{Z}}\left[g_{\mathrm{relu}}(\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  together with  $-\mathbb{E}_{\mathbf{Z}}\left[\frac{\partial\ell}{\partial\boldsymbol{v}} (\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  form a descent direction for minimizing the population loss.

Lemma 5. If  $\pmb{w} \neq \mathbf{0}_n$  and  $\theta(\pmb{w}, \pmb{w}^*) \in (0, \pi)$ , then the inner product between the expected coarse and true gradients w.r.t.  $\pmb{w}$  is

$$
\left\langle \mathbb {E} _ {\mathbf {Z}} \left[ g _ {\text {r e l u}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right], \frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}) \right\rangle = \frac {\sin \left(\theta (\boldsymbol {w} , \boldsymbol {w} ^ {*})\right)}{2 (\sqrt {2 \pi}) ^ {3} \| \boldsymbol {w} \|} \left(\boldsymbol {v} ^ {\top} \boldsymbol {v} ^ {*}\right) ^ {2} \geq 0.
$$

Moreover, if further  $\| \pmb{v} \| \leq C_{\pmb{v}}$ , there exists a constant  $A > 0$  depending only on  $C_{\pmb{v}}$ , such that

$$
\left\| \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\text {r e l u}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right] \right\| ^ {2} \leq A \left(\left\| \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v}, \boldsymbol {w}) \right\| ^ {2} + \left\langle \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\text {r e l u}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right], \frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}) \right\rangle\right). \tag {12}
$$

Clearly, when  $\left\langle \mathbb{E}_{\mathbf{Z}}\left[g_{\mathrm{relu}}(\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right],\frac{\partial f}{\partial\boldsymbol{w}} (\boldsymbol {v},\boldsymbol {w})\right\rangle >0,\mathbb{E}_{\mathbf{Z}}\left[g_{\mathrm{relu}}(\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  is roughly in the same direction as  $\frac{\partial f}{\partial w} (\boldsymbol {v},\boldsymbol {w})$  . Moreover, since by Lemma 4,  $\mathbb{E}_{\mathbf{Z}}\left[\frac{\partial\ell}{\partial v} (\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right] = \frac{\partial f}{\partial v} (\boldsymbol {v},\boldsymbol {w})$  , we expect that the coarse gradient descent behaves like the gradient descent directly on  $f(\boldsymbol {v},\boldsymbol {w})$  . We would like to highlight the significance of the estimate (12) in guaranteeing the descent property of Algorithm 1. By the Lipschitz continuity of  $\nabla f$  specified in Lemma 3, it holds that

$$
\begin{array}{l} f (\boldsymbol {v} ^ {t + 1}, \boldsymbol {w} ^ {t + 1}) - f (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}) \leq \left\langle \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}), \boldsymbol {v} ^ {t + 1} - \boldsymbol {v} ^ {t} \right\rangle + \left\langle \frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}), \boldsymbol {w} ^ {t + 1} - \boldsymbol {w} ^ {t} \right\rangle \\ + \frac {L}{2} \left(\| \boldsymbol {v} ^ {t + 1} - \boldsymbol {v} ^ {t} \| ^ {2} + \| \boldsymbol {w} ^ {t + 1} - \boldsymbol {w} ^ {t} \| ^ {2}\right) \\ = - \left(\eta - \frac {L \eta^ {2}}{2}\right) \left\| \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}) \right\| ^ {2} + \frac {L \eta^ {2}}{2} \left\| \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\text {r e l u}} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}; \mathbf {Z}) \right] \right\| ^ {2} \\ \left. - \eta \left\langle \frac {\partial f}{\partial \boldsymbol {w}} \left(\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}\right), \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\text {r e l u}} \left(\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}; \mathbf {Z}\right) \right] \right\rangle \right. \\ \stackrel {a)} {\leq} - \left(\eta - (1 + A) \frac {L \eta^ {2}}{2}\right) \left\| \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}) \right\| ^ {2} \\ - \left(\eta - \frac {A L \eta^ {2}}{2}\right) \left\langle \frac {\partial f}{\partial \boldsymbol {w}} \left(\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}\right), \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\text {r e l u}} \left(\boldsymbol {v} ^ {t}, \boldsymbol {w} ^ {t}; \mathbf {Z}\right) \right] \right\rangle , \tag {13} \\ \end{array}
$$

where a) is due to (12). Therefore, if  $\eta$  is small enough, we have monotonically decreasing population loss until convergence.

Lemma 6. When Algorithm 1 converges,  $\mathbb{E}_{\mathbf{Z}}\left[\frac{\partial\ell}{\partial\boldsymbol{v}} (\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  and  $\mathbb{E}_{\mathbf{Z}}\left[g_{\mathrm{relu}}(\boldsymbol {v},\boldsymbol {w};\mathbf{Z})\right]$  vanish simultaneously, which only occurs at the

1. Saddle points where (8) is satisfied according to Proposition 1.  
2. Minimizers of (2) where  $\pmb{v} = \pmb{v}^{*}$ ,  $\theta(\pmb{w}, \pmb{w}^{*}) = 0$ , or  $\pmb{v} = (\pmb{I}_m + \pmb{1}_m \pmb{1}_m^\top)^{-1} (\pmb{1}_m \pmb{1}_m^\top - \pmb{I}_m) \pmb{v}^*$ ,  $\theta(\pmb{w}, \pmb{w}^{*}) = \pi$ .

# 3.2 DERIVATIVE OF THE IDENTITY FUNCTION AS STE

Now we consider the derivative of identity function. As opposed to the ReLU case, similar results to Lemmas 5 and 6 are not valid anymore. It happens that the coarse gradient derived from the identity STE does not vanish at the local minimum, and Algorithm 1 may never converge there.

Lemma 7. Let  $\mu (x) = x$  in (5). Then the expected coarse partial gradient w.r.t.  $\mathbf{w}$  is

$$
\mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\mathrm {i d}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right] = \frac {1}{\sqrt {2 \pi}} \left(\| \boldsymbol {v} \| ^ {2} \frac {\boldsymbol {w}}{\| \boldsymbol {w} \|} - \left(\boldsymbol {v} ^ {\top} \boldsymbol {v} ^ {*}\right) \boldsymbol {w} ^ {*}\right). \tag {14}
$$

If  $\theta (\pmb {w},\pmb{w}^{*}) = \pi$  and  $\pmb {v} = \left(\pmb {I}_m + \pmb {1}_m\pmb {1}_m^\top\right)^{-1}(\pmb {1}_m\pmb {1}_m^\top -\pmb {I}_m)\pmb{v}^*$

$$
\left\| \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\mathrm {i d}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right] \right\| = \frac {2 (m - 1)}{\sqrt {2 \pi} (m + 1) ^ {2}} \left(\mathbf {1} _ {m} ^ {\top} \boldsymbol {v} ^ {*}\right) ^ {2} \geq 0,
$$

i.e.,  $\mathbb{E}_{\mathbf{Z}}\left[\pmb{g}_{\mathrm{id}}(\pmb{v},\pmb{w};\mathbf{Z})\right]$  does not vanish at the spurious local minimizers if  $\mathbf{1}_m^\top \pmb{v}^* \neq 0$  and  $m > 1$ .

Lemma 8. If  $\pmb{w} \neq \mathbf{0}_n$  and  $\theta(\pmb{w}, \pmb{w}^*) \in (0, \pi)$ , then the inner product between the expected coarse and true gradients w.r.t.  $\pmb{w}$  is

$$
\left\langle \mathbb {E} _ {\mathbf {Z}} \left[ \boldsymbol {g} _ {\mathrm {i d}} (\boldsymbol {v}, \boldsymbol {w}; \mathbf {Z}) \right], \frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v}, \boldsymbol {w}) \right\rangle = \frac {\sin (\theta (\boldsymbol {w} , \boldsymbol {w} ^ {*}))}{(\sqrt {2 \pi}) ^ {3} \| \boldsymbol {w} \|} (\boldsymbol {v} ^ {\top} \boldsymbol {v} ^ {*}) ^ {2} \geq 0. \tag {15}
$$

When  $\theta (\pmb {w},\pmb{w}^{*})\to \pi$ $\pmb {v}\rightarrow \left(\pmb {I}_m + \pmb {1}_m\pmb {1}_m^\top\right)^{-1}(\pmb {1}_m\pmb {1}_m^\top -\pmb {I}_m)\pmb {v}^*$  , if  $\mathbf{1}_m^\top \pmb {v}^*\neq 0$  and  $m > 1$  , we have

$$
\frac {\left\| \mathbb {E} _ {\mathbf {Z}} \left[ g _ {\mathrm {i d}} (\boldsymbol {v} , \boldsymbol {w} ; \mathbf {Z}) \right]\right\| ^ {2}}{\left\| \frac {\partial f}{\partial \boldsymbol {v}} (\boldsymbol {v} , \boldsymbol {w}) \right\| ^ {2} + \left\langle \mathbb {E} _ {\mathbf {Z}} \left[ g _ {\mathrm {i d}} (\boldsymbol {v} , \boldsymbol {w}; \mathbf {Z}) \right], \frac {\partial f}{\partial \boldsymbol {w}} (\boldsymbol {v} , \boldsymbol {w}) \right\rangle} \rightarrow + \infty . \tag {16}
$$

Lemma 7 suggests that if  $\mathbf{1}_m^\top \pmb{v}^* \neq 0$ , the coarse gradient descent will never converge near the spurious minimizers with  $\theta(\pmb{w}, \pmb{w}^*) = \pi$  and  $\pmb{v} = (I_m + \mathbf{1}_m \mathbf{1}_m^\top)^{-1} (\mathbf{1}_m \mathbf{1}_m^\top - \mathbf{I}_m) \pmb{v}^*$ , because  $\mathbb{E}_{\mathbf{Z}}[g_{\mathrm{id}}(\pmb{v}, \pmb{w}; \mathbf{Z})]$  does not vanish there. By the positive correlation implied by (15), for some proper  $(\pmb{v}^0, \pmb{w}^0)$ , the iterates  $\{(v^t, w^t)\}$  may move towards a spurious local minimizer in the beginning. But when  $\{(v^t, w^t)\}$  approaches it, the correlation in (15) goes to 0, and the descent property (13) does not hold with  $\mathbb{E}_{\mathbf{Z}}[g_{\mathrm{id}}(\pmb{v}, \pmb{w}; \mathbf{Z})]$  because of (16), hence the instability arises.

# 4 EXPERIMENTS

In practice, the vanilla ReLU may not deliver the best empirical performance. Compared with the identity function and vanilla ReLU, the clipped ReLU proposed in (Cai et al., 2017) approximates the quantized ReLU better. The plots of the 2-bit quantized ReLU and its associated clipped ReLU are in Figure 3 in the appendix. Besides the two STEs discussed above, we include the STE using derivative of the clipped ReLU for comparisons on training 2-bit and 4-bit activation networks for MNIST (LeCun et al., 1998) and CIFAR-10 (Krizhevsky, 2009) classifications. Log-tailed ReLU has also been proposed, we do not consider it here since it gives similar performance to the clipped ReLU as reported in (Cai et al., 2017). We emphasize that we are not claiming the superiority of the quantization approach used here, as it is nothing but the HWGQ (Cai et al., 2017), except we consider the uniform quantization. In all of our experiments, the weights are kept float.

The optimizer we use is the stochastic (coarse) gradient descent with momentum  $= 0.9$  for all experiments. We train 50 epochs for LeNet-5 (LeCun et al., 1998) on MNIST, and 200 epochs for VGG-11 (Simonyan & Zisserman, 2014) and ResNet-20 (He et al., 2016) on CIFAR-10. The parameters/weights are initialized with those from their pre-trained full-precision counterparts. The schedule of the learning rate is specified in Table 2 in the appendix.

The resolution  $\alpha$  for the quantized ReLU needs to be carefully chosen to maintain the full-precision level accuracy. To this end, we follow (Cai et al., 2017) and resort to a modified batch normalization layer (Ioffe & Szegedy, 2015) without the scale and shift, whose output components approximately follow a Gaussian distribution. Then the  $\alpha$  that fits the layer input the best can be pre-computed by a variant of Lloyd's algorithm (Lloyd, 1982; Yin et al., 2018) applied to a set of simulated 1-D half-Gaussian data. After determining the  $\alpha$ , it will be fixed during the whole training process. Since the original LeNet-5 does not have batch normalization, we add one prior to each activation layer.

# 4.1 COMPARISON RESULTS

The experimental results are summarized in Table 1, where we record both the training losses and validation accuracies. Among the three STEs, the derivative of clipped ReLU gives the best overall performance, followed by vanilla ReLU and then by the identity function. For deeper networks, clipped ReLU is the best performer. But on the shallow LeNet-5 network, vanilla ReLU exhibits comparable performance to clipped ReLU, which is in line with our theoretical finding that ReLU is a superior STE for learning the one-hidden-layer (shallow) CNN.

# 4.2 REPELLED FROM IMPROVED MINIMA

We report the phenomenon of being repelled from a good minimum on ResNet-20 with 4-bit activations when using the identity STE. By Table 1, the coarse gradient descent algorithms using the

<table><tr><td rowspan="2"></td><td rowspan="2">Network</td><td rowspan="2">BitWidth</td><td colspan="3">Straight-through estimator</td></tr><tr><td>identity</td><td>vanilla ReLU</td><td>clipped ReLU</td></tr><tr><td rowspan="2">MNIST</td><td rowspan="2">LeNet5</td><td>2</td><td>2.6 × 10-2/98.49</td><td>5.1 × 10-3/99.24</td><td>5.4 × 10-3/99.23</td></tr><tr><td>4</td><td>6.0 × 10-3/98.98</td><td>9.0 × 10-4/99.32</td><td>8.8 × 10-4/99.24</td></tr><tr><td rowspan="4">CIFAR10</td><td rowspan="2">VGG11</td><td>2</td><td>0.19/86.58</td><td>0.10/88.69</td><td>0.02/90.92</td></tr><tr><td>4</td><td>3.1 × 10-2/90.19</td><td>1.5 × 10-3/92.01</td><td>1.3 × 10-3/92.08</td></tr><tr><td rowspan="2">ResNet20</td><td>2</td><td>1.56/46.52</td><td>1.50/48.05</td><td>0.24/88.39</td></tr><tr><td>4</td><td>1.38/54.16</td><td>0.25/86.59</td><td>0.04/91.24</td></tr></table>

Table 1: Training loss/Validation accuracy (%) on MNIST and CIFAR-10 with quantized activations and float weights, for STEs using derivatives of the identity function, vanilla ReLU and clipped ReLU at bit-widths 2 and 4.

vanilla and clipped ReLUs converge to the neighborhoods of the minima with validation accuracies (training losses) of  $86.59\%$  (0.25) and  $91.24\%$  (0.04), respectively, whereas that using the identity STE gives  $54.16\%$  (1.38). Note that the landscape of the empirical loss function does not depend on which STE is used in the training. Then we initialize training with the two improved minima and use the identity STE. To see if the algorithm is stable there, we start the training with a tiny learning rate of  $10^{-5}$ . For both initializations, the training loss and validation error significantly increase within the first 20 epochs. At epoch 20, we switch to the normal schedule of learning rate and run 200 additional epochs. The training using the identity STE ends up with a much worse minimum. This is because the coarse gradient with identity STE does not vanish at the good minima in this case.

![](images/238f8b1ad24995ede34dfae8e68870b48cf0ef15393261f6d375089a620f3b7d.jpg)  
Figure 2: When initialized with the improved minima produced by the vanilla (orange) and clipped (blue) ReLUs on ResNet-20 with 4-bit activations and float weights, the coarse gradient descent using the identity STE ends up being repelled from there. The learning rate is set to  $10^{-5}$  until epoch 20.

![](images/f98f94966d883866ef70e92500674e1c71feafd5067dd1435b4d016be45f81c4.jpg)

# 5 CONCLUDING REMARKS

We provided the first theoretical justification for the concept of STE. We considered two STEs: the derivatives of the identity function and the vanilla ReLU, in training one-hidden-layer CNN with binary activation. We derived the explicit formulas of the expected coarse gradients corresponding to the two STEs, and showed that the negative expected coarse gradient based on vanilla ReLU is a descent direction for minimizing the population loss, whereas the identity STE is not. Our experiments on MNIST and CIFAR-10 datasets verified the theoretical findings. Looking ahead, we believe that the closeness between the quantized ReLU and the anti-derivative of STE plays an essential role in the performance of the coarse gradient descent. Hence, we would like to explore and quantify this relationship in the future work.

# ACKNOWLEDGMENTS

Use unnumbered third level headings for the acknowledgments. All acknowledgments, including those to funding agencies, go at the end of the paper.

# REFERENCES

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.  
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Alon Brutzkus and Amir Globerson. Globally optimal gradient descent for a convnet with gaussian inputs. arXiv preprint arXiv:1702.07966, 2017.  
Zhaowei Cai, Xiaodong He, Jian Sun, and Nuno Vasconcelos. Deep learning with low precision by half-wave gaussian quantization. In IEEE Conference on Computer Vision and Pattern Recognition, 2017.  
Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks. arXiv preprint arXiv:1805.06085, 2018.  
Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In International Conference on Machine Learning, pp. 160-167. ACM, 2008.  
Matthieu Courbariaux, Yoshua Bengio, and Jean-Pierre David. Binaryconnect: Training deep neural networks with binary weights during propagations. In Advances in neural information processing systems, pp. 3123-3131, 2015.  
Simon S. Du, Jason D. Lee, Yuandong Tian, Barnabas Poczos, and Aarti Singh. Gradient descent learns one-hidden-layer cnn: Don't be afraid of spurious local minimum. arXiv preprint arXiv:1712.00779, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Geoffrey Hinton. Neural networks for machine learning, coursera. Coursera, video lectures, 2012.  
Lu Hou and James T Kwok. Loss-aware weight quantization of deep networks. arXiv preprint arXiv:1802.08635, 2018.  
Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks: Training neural networks with weights and activations constrained to +1 or -1. arXiv preprint arXiv:1602.02830, 2016.  
Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Quantized neural networks: Training neural networks with low precision weights and activations. Journal of Machine Learning Research, 18:1-30, 2018.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. Tech Report, 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Fengfu Li, Bo Zhang, and Bin Liu. Ternary weight networks. arXiv preprint arXiv:1605.04711, 2016.

Hao Li, Soham De, Zheng Xu, Christoph Studer, Hanan Samet, and Tom Goldstein. Training quantized nets: A deeper understanding. In Advances in Neural Information Processing Systems, pp. 5811-5821, 2017.  
Yuanzhi Li and Yang Yuan. Convergence analysis of two-layer neural networks with relu activation. In Advances in Neural Information Processing Systems, pp. 597-607, 2017.  
Stuart P. Lloyd. Least squares quantization in pmc. IEEE Trans. Info. Theory, 28:129-137, 1982.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. In European Conference on Computer Vision, pp. 525-542. Springer, 2016.  
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, pp. 91-99, 2015.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484, 2016.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Mahdi Soltanolkotabi. Learning relus via gradient descent. In Advances in Neural Information Processing Systems, pp. 2007-2017, 2017.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
Yuandong Tian. An analytical formula of population gradient for two-layered relu network and its applications in convergence and critical point analysis. arXiv preprint arXiv:1703.00560, 2017.  
Bao Wang, Xiyang Luo, Zhen Li, Wei Zhu, Zuoqiang Shi, and Stanley J Osher. Deep neural nets with interpolating function as output activation. In Advances in Neural Information Processing Systems, 2018.  
Penghang Yin, Shuai Zhang, Jiancheng Lyu, Stanley Osher, Yingyong Qi, and Jack Xin. Binary relax: A relaxation approach for training deep neural networks with quantized weights. arXiv preprint arXiv:1801.06313, 2018.  
Kai Zhong, Zhao Song, Prateek Jain, Peter L Bartlett, and Inderjit S Dhillon. Recovery guarantees for one-hidden-layer neural networks. arXiv preprint arXiv:1706.03175, 2017.  
Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, and Yurong Chen. Incremental network quantization: Towards lossless cnns with low-precision weights. arXiv preprint arXiv:1702.03044, 2017.  
Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, and Yuheng Zou. Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint arXiv:1606.06160, 2016.  
Chenzhuo Zhu, Song Han, Huizi Mao, and William J Dally. Trained ternary quantization. arXiv preprint arXiv:1612.01064, 2016.
