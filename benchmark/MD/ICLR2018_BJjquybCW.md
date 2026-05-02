# THE LOSS SURFACE AND EXPRESSIVITY OF DEEP CONVOLUTIONAL NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We analyze the expressiveness and loss surface of practical deep convolutional neural networks (CNNs) with shared weights and max pooling layers. We show that such CNNs produce linearly independent features at a "wide" layer which has more neurons than the number of training samples. This condition holds e.g. for the VGG network. Furthermore, we provide for such wide CNNs necessary and sufficient conditions for global minima with zero training error. For the case where the wide layer is followed by a fully connected layer we show that almost every critical point of the empirical loss is a global minimum with zero training error. Our analysis suggests that both depth and width are very important in deep learning. While depth brings more representational power and allows the network to learn high level features, width smoothes the optimization landscape of the loss function in the sense that a sufficiently wide network has a well-behaved loss surface with almost no bad local minima.

# 1 INTRODUCTION

It is well known that the optimization problem for training neural networks can have exponentially many local minima (Auer et al., 1996; Safran & Shamir, 2016) and NP-hardness has been shown in many cases (Blum & Rivest., 1989; Sima, 2002; Livni et al., 2014; Shamir, 2017; Shalev-Shwartz et al., 2017). However, it has been empirically observed (Dauphin et al., 2014; Goodfellow et al., 2015) that the training of state-of-the-art deep CNNs (LeCun et al., 1990; Krizhevsky et al., 2012), which are often overparameterized, is not hampered by suboptimal local minima.

In order to explain the apparent gap between hardness results and practical performance, many interesting theoretical results have been recently developed (Andoni et al., 2014; Sedghi & Anandkumar, 2015; Janzamin et al., 2016; Haeffele & Vidal, 2015; Gautier et al., 2016; Brutzkus & Globerson, 2017; Soltanolkotabi, 2017; Soudry & Hoffer, 2017; Goel & Klivans, 2017; Du et al., 2017; Zhong et al., 2017; Tian, 2017; Li & Yuan, 2017) in order to identify conditions under which one can guarantee that local search algorithms like gradient descent converge to the globally optimal solution. However, it turns out that these approaches are either not practical as they require e.g. knowledge about the data generating measure, or a modification of network structure and objective, or they are for quite restricted network structures, mostly one hidden layer networks, and thus are not able to explain the success of deep networks in general. For deep linear networks one has achieved a quite complete picture of the loss surface as it has been shown that every local minimum is a global minimum (Baldi & Hornik, 1988; Kawaguchi, 2016; Freeman & Bruna, 2017; Hardt & Ma, 2017; Yun et al., 2017). By randomizing the nonlinear part of a feedforward network with ReLU activation function and making some additional simplifying assumptions, Choromanska et al. (2015a) can relate the loss surface of neural networks to a certain spin glass model. In this model the objective of local minima is close to the global optimum and the number of bad local minima decreases quickly with the distance to the global optimum. This is a very interesting result but is based on a number of unrealistic assumptions (Choromanska et al., 2015b). More recently, Nguyen & Hein (2017) have analyzed deep fully connected networks with general activation functions and could show that almost every critical point is a global minimum if one layer has more neurons than the number of training points. While this result holds for networks in practice, it requires a quite extensively overparameterized network.

In this paper we overcome the restriction of previous work in several ways. This paper is one of the first ones, which studies CNNs. CNNs are of high practical interest as they learn very useful representations (Zeiler & Fergus, 2014; Mahendran & Vedaldi, 2015; Yosinski et al., 2015) with a small number of parameters. We are only aware of Cohen & Shashua (2016) who study the expressiveness of CNNs with max-pooling layer and ReLU activation but with rather unrealistic filters (just  $1 \times 1$ ) and no shared weights. In our setting we allow as well max pooling and general activation functions. Moreover, we can have an arbitrary number of filters and we study general convolutions as the filters need not be applied to regular structures like  $3 \times 3$  but can be patch-based where the only condition is that all the patches have the size of the filter. Convolutional layers, fully connected layers and max-pooling layers can be combined in almost arbitrary order. We study in this paper the expressiveness and loss surface of such a CNN where one layer is wide, in the sense that it has more neurons than the number of training points. While this assumption sounds at first quite strong, we want to emphasize that the VGG network (Simonyan & Zisserman, 2015) and other CNNs, see Table 1, fulfill this condition. We show that such wide CNNs produce linearly independent feature representations at the wide layer and thus are able to fit the training data exactly (universal finite sample expressivity). This is even true if the bottom layers (from input to the wide layer) are chosen randomly with probability one. We think that this explains partially the results of Zhang et al. (2017) where they show experimentally for several CNNs that they are able to fit random labels. Moreover, we provide necessary and sufficient conditions for global minima with zero squared loss and show for a particular class of CNNs that almost all critical points are globally optimal, which to some extent explains why such wide CNNs can be optimized so efficiently.

# 2 DEEP CONVOLUTIONAL NEURAL NETWORKS

We first introduce our notation and definition of CNNs. Let  $N$  be the number of training samples and denote by  $X = [x_{1},\ldots ,x_{N}]^{T}\in \mathbb{R}^{N\times d}$ ,  $Y = [y_{1},\ldots ,y_{N}]^{T}\in \mathbb{R}^{N\times m}$  the input resp. output matrix for the training data  $(x_{i},y_{i})_{i = 1}^{N}$ , where  $d$  is the input dimension and  $m$  the number of classes.

Let  $L$  be the number of layers of the network, where each layer is either a convolutional, max-pooling or fully connected layer. The layers are indexed from  $k = 0,1,\dots ,L$  which corresponds to input layer, 1st hidden layer, ..., and output layer. Let  $n_k$  be the width of layer  $k$  and  $f_{k}:\mathbb{R}^{d}\to \mathbb{R}^{n_{k}}$  the function that computes for every input its feature vector at layer  $k$ .

The convolutional layer consists of a set of patches of equal length where every patch is a subset of neurons from the same layer. Throughout this paper, we assume that the patches of every layer cover the whole layer, i.e. every neuron belongs to at least one of the patches, and that there are no patches that contain exactly the same subset of neurons. This means that if one patch covers the whole layer then it must be the only patch of the layer. Let  $P_{k}$  and  $l_{k}$  be the number of patches resp. the size of each patch at layer  $k$  for every  $0 \leq k < L$ . For every input  $x \in \mathbb{R}^{d}$ , let  $\{x^{1},\ldots ,x^{P_{0}}\} \in \mathbb{R}^{l_{0}}$  denote the set of patches at the input layer and  $\left\{f_k^1 (x),\dots ,f_k^{P_k}(x)\right\} \in \mathbb{R}^{l_k}$  the set of patches at layer  $k$ . Each filter of the layer consists of the same set of patches. We denote by  $T_{k}$  the number of convolutional filters and by  $W_{k} = [w_{k}^{1},\dots ,w_{k}^{T_{k}}] \in \mathbb{R}^{l_{k - 1}\times T_{k}}$  the corresponding parameter matrix of the convolutional layer  $k$  for every  $1 \leq k < L$ . Each column of  $W_{k}$  corresponds to one filter. Furthermore,  $b_{k} \in \mathbb{R}^{n_{k}}$  denotes the bias vector and  $\sigma_{k}: \mathbb{R} \to \mathbb{R}$  the activation function for each layer. Note that one can use the same activation function for all layers but we use the general form to highlight the role of different layers. In this paper, all functions are applied componentwise, and we denote by  $[a]$  the set of integers  $\{1,2,\dots ,a\}$  and by  $[a,b]$  the set of integers from  $a$  to  $b$ .

Definition 2.1 (Convolutional layer) A layer  $k$  is called a convolutional layer if its output  $f_{k}(x) \in \mathbb{R}^{n_{k}}$  is defined for every  $x \in \mathbb{R}^d$  as

$$
f _ {k} (x) _ {h} = \sigma_ {k} \left(\left\langle w _ {k} ^ {t}, f _ {k - 1} ^ {p} (x) \right\rangle + \left(b _ {k}\right) _ {h}\right) \tag {1}
$$

for every  $p\in [P_{k - 1}],t\in [T_k],h\coloneqq (p - 1)T_k + t$

The value of each neuron at layer  $k$  is computed by first taking the inner product between a filter of layer  $k$  and a patch at layer  $k - 1$ , adding the bias and then applying the activation function. The number of neurons at layer  $k$  is thus  $n_k = T_kP_{k - 1}$ , which we denote as the width of layer  $k$ .

Our definition of a convolutional layer is quite general as every patch can be an arbitrary subset of neurons of the same layer and thus covers most of existing variants in practice.

Definition 2.1 includes the fully connected layer as a special case by using  $P_{k-1} = 1, l_{k-1} = n_{k-1}, f_{k-1}^{1}(x) = f_{k-1}(x) \in \mathbb{R}^{n_{k-1}}, T_{k} = n_{k}, W_{k} \in \mathbb{R}^{n_{k-1} \times n_{k}}, b_{k} \in \mathbb{R}^{n_{k}}$ . Thus we have only one patch which is the whole feature vector at this layer.

Definition 2.2 (Fully connected layer) A layer  $k$  is called a fully connected layer if its output  $f_{k}(x)\in \mathbb{R}^{n_{k}}$  is defined for every  $x\in \mathbb{R}^d$  as

$$
f _ {k} (x) = \sigma_ {k} \left(W _ {k} ^ {T} f _ {k - 1} (x) + b _ {k}\right). \tag {2}
$$

For some results we also allow max-pooling layers.

Definition 2.3 (Max-pooling layer) A layer  $k$  is called a max-pooling layer if its output  $f_{k}(x) \in \mathbb{R}^{n_{k}}$  is defined for every  $x \in \mathbb{R}^d$  and  $p \in [P_{k - 1}]$  as

$$
f _ {k} (x) _ {p} = \max  \left(\left(f _ {k - 1} ^ {p} (x)\right) _ {1}, \dots , \left(f _ {k - 1} ^ {p} (x)\right) _ {l _ {k - 1}}\right). \tag {3}
$$

A max-pooling layer just computes the maximum element of every patch from the previous layer. Since there are  $P_{k-1}$  patches at layer  $k-1$ , the number of output neurons at layer  $k$  is  $n_k = P_{k-1}$ .

Reformulation of Convolutional Layers: For each convolutional or fully connected layer, we denote by  $\mathcal{M}_k:\mathbb{R}^{l_{k - 1}\times T_k}\to \mathbb{R}^{n_{k - 1}\times n_k}$  the linear map that returns for every parameter matrix  $W_{k}\in \mathbb{R}^{l_{k - 1}\times T_{k}}$  the corresponding full weight matrix  $U_{k} = \mathcal{M}_{k}(W_{k})\in \mathbb{R}^{n_{k - 1}\times n_{k}}$ . For convolutional layers,  $U_{k}$  can be seen as the counterpart of the weight matrix  $W_{k}$  in fully connected layers. We define  $U_{k} = \mathcal{M}_{k}(W_{k}) = W_{k}$  if layer  $k$  is fully connected. Note that the mapping  $\mathcal{M}_k$  depends on the patch structure of each convolutional layer  $k$ . For example, suppose that layer  $k$  has two filters of length 3, that is,  $W_{k} = [w_{k}^{1},w_{k}^{2}] = \begin{bmatrix} a & d\\ b & e\\ c & f \end{bmatrix}$ , and  $n_{k - 1} = 5$  and patches given by

a 1D-convolution with stride 1 and no padding then:  $U_{k}^{T} = \mathcal{M}_{k}(W_{k})^{T} = \begin{bmatrix} a & b & c & 0 & 0 \\ d & e & f & 0 & 0 \\ 0 & a & b & c & 0 \\ 0 & d & e & f & 0 \\ 0 & 0 & a & b & c \\ 0 & 0 & d & e & f \end{bmatrix}$ ,

The above ordering of the rows of  $U_{k}^{T}$  of a convolutional layer is determined by Equation (1), in particular, the row index  $h$  of  $U_{k}^{T}$  is calculated as  $h = (p - 1)T_{k} + t$ , which means for every given patch  $p$  one has to loop over all the filters  $t$  and compute the corresponding value of the output unit by taking the inner product of the  $h$ -th row of  $U_{k}^{T}$  with the whole feature vector of the previous layer.

We assume throughout this paper that there is no non-linearity at the output layer. By ignoring max-pooling layers for the moment, the feature maps  $f_{k}:\mathbb{R}^{d}\to \mathbb{R}^{n_{k}}$  can be written as

$$
f _ {0} (x) = x, \quad f _ {k} (x) = \sigma_ {k} \left(g _ {k} (x)\right), \text {w h e r e} g _ {k} (x) = U _ {k} ^ {T} f _ {k - 1} (x) + b _ {k}, \quad \forall 1 \leq k \leq L - 1
$$

$$
f _ {L} (x) = g _ {L} (x) = U _ {L} ^ {T} f _ {L - 1} (x) + b _ {L},
$$

where  $g_{k}:\mathbb{R}^{d}\to \mathbb{R}^{n_{k}}$  is the pre-activation output at layer  $k$ . By stacking the feature vectors of layer  $k$  of all training samples, before and after applying the activation function, into a matrix, we define:

$$
F _ {k} = \left[ f _ {k} \left(x _ {1}\right), \dots , f _ {k} \left(x _ {N}\right) \right] ^ {T} \in \mathbb {R} ^ {N \times n _ {k}}, \quad \text {a n d} \quad G _ {k} = \left[ g _ {k} \left(x _ {1}\right), \dots , g _ {k} \left(x _ {N}\right) \right] ^ {T} \in \mathbb {R} ^ {N \times n _ {k}}.
$$

In this paper, we refer to  $F_{k}$  as the output matrix at layer  $k$ . It follows from above that

$$
F _ {0} = X, \quad F _ {k} = \sigma_ {k} \left(G _ {k}\right), \text {w h e r e} G _ {k} = F _ {k - 1} U _ {k} + \mathbf {1} _ {N} b _ {k} ^ {T}, \quad \forall 1 \leq k \leq L - 1 \tag {4}
$$

$$
F _ {L} = G _ {L} = F _ {L - 1} U _ {L} + \mathbf {1} _ {N} b _ {L} ^ {T}. \tag {5}
$$

In this paper, we assume the following general condition on the structure of convolutional layers.

Assumption 2.4 (Convolutional Structure) For every convolutional layer  $k$ , there exists at least one parameter matrix  $W_{k} \in \mathbb{R}^{l_{k-1} \times T_{k}}$  for which the corresponding weight matrix  $U_{k} = \mathcal{M}_{k}(W_{k}) \in \mathbb{R}^{n_{k-1} \times n_{k}}$  has full rank.

It is straightforward to see that Assumption 2.4 is satisfied if every neuron of a convolutional layer belongs to at least one patch and there are no identical patches.

Lemma 2.5 If Assumption 2.4 holds, then for every convolutional layer  $k$ , the set of  $W_{k} \in \mathbb{R}^{l_{k-1} \times T_{k}}$  for which  $U_{k} = \mathcal{M}_{k}(W_{k}) \in \mathbb{R}^{n_{k-1} \times n_{k}}$  does not have full rank has Lebesgue measure zero.

Proof: Since  $U_{k} = \mathcal{M}_{k}(W_{k})\in \mathbb{R}^{n_{k - 1}\times n_{k}}$  and  $\mathcal{M}_k$  is a linear map, every entry of  $U_{k}$  is a linear function of the entries of  $W_{k}$ . Let  $m = \min (n_{k - 1},n_k)$ , then the set of low rank matrices  $U_{k}$  is characterized by a system of equations where the  $\binom{\max(n_{k-1},n_k)}{m}$  determinants of all  $m\times m$  sub-matrices of  $U_{k}$  are zero. As the determinant is a polynomial in the entries of the matrix and thus a real analytic function, and the composition of analytic functions is again analytic, we get that each determinant is a real analytic function of  $W_{k}$ . By Assumption 2.4, there exists at least one  $W_{k}$  such that one of these determinants is non-zero. Thus by Lemma A.2, the set of  $W_{k}$  where this determinant is zero has Lebesgue measure zero. As all the submatrices need to have low rank in order that  $U_{k}$  has low rank, we get that the set of  $W_{k}$  where  $U_{k}$  has low rank has Lebesgue measure zero.

# 3 WIDE CNNS CAN LEARN LINEARLY INDEPENDENT FEATURES

In this section, we show that a class of standard CNN architectures with convolutional layers, fully connected layers and max-pooling layers plus standard activation functions like ReLU, sigmoid, softmax, etc are able to learn linearly independent features at any hidden layer if that layer has more neurons than the number of training samples. Our assumption on training data is the following.

Assumption 3.1 (Training data) The patches of different training samples are non-identical, that is,  $x_{i}^{p} \neq x_{j}^{q}$  for every  $p, q \in [P_0], i, j \in [N], i \neq j$ .

Assumption 3.1 is quite weak, especially if the size of the input patches is larger. If the assumption does not hold, one can add a small perturbation to the training samples:  $\{x_{1} + \epsilon_{1},\ldots ,x_{N} + \epsilon_{N}\}$ . The set of  $\{\epsilon_i\}_{i = 1}^N$  where Assumption 3.1 is not fulfilled for the new dataset has measure zero. Moreover,  $\{\epsilon_i\}_{i = 1}^N$  can be chosen arbitrarily small so that the influence of the perturbation is negligible. Our main assumptions on the activation function of the hidden layers is the following.

Assumption 3.2 (Activation function) The activation function  $\sigma$  is continuous, non-constant, and satisfies one of the following conditions:

-  $\exists \mu_{+}, \mu_{-} \in \mathbb{R}$  s.t.  $\lim_{t \to -\infty} \sigma_{k}(t) = \mu_{-}$  and  $\lim_{t \to \infty} \sigma_{k}(t) = \mu_{+}$  and  $\mu_{+} \mu_{-} = 0$  
-  $\exists \rho_{1}, \rho_{2}, \rho_{3}, \rho_{4} \in \mathbb{R}_{+}$  s.t.  $|\sigma(t)| \leq \rho_{1} e^{\rho_{2} t}$  for  $t < 0$  and  $|\sigma(t)| \leq \rho_{3} t + \rho_{4}$  for  $t \geq 0$ .

Assumption 3.2 covers several standard activation functions.

Lemma 3.3 The following activation functions satisfy Assumption 3.2:

ReLU:  $\sigma(t) = \max(0, t)$ , Sigmoid:  $\sigma(t) = \frac{1}{1 + e^{-t}}$ , Softplus:  $\sigma_{\alpha}(t) = \frac{1}{\alpha} \ln (1 + e^{\alpha t})$  for  $\alpha > 0$ .

The softplus function is a smooth approximation of ReLU. It holds:

$$
\lim  _ {\alpha \rightarrow \infty} \sigma_ {\alpha} (t) = \lim  _ {\alpha \rightarrow \infty} \frac {1}{\alpha} \ln (1 + e ^ {\alpha t}) = \max  (0, t). \tag {6}
$$

The first main result of this paper is the following.

Theorem 3.4 (Linearly Independent Features) Let Assumption 3.1 hold for the training sample. Consider a deep CNN architecture for which there exists some layer  $1 \leq k \leq L - 1$  such that

1. Layer 1 and layer  $k$  are convolutional or fully connected while all the other layers can be convolutional, fully connected or max-pooling  
2. The width of layer  $k$  is larger than the number of training samples,  $n_k = T_k P_{k-1} \geq N$  
3.  $(\sigma_{1},\ldots ,\sigma_{k})$  satisfy Assumption 3.2

Then there exist a set of parameters of the first  $k$  layers  $(W_{l},b_{l})_{l = 1}^{k}$  such that the set of feature vectors  $\{f_k(x_1),\ldots ,f_k(x_N)\}$  are linearly independent. Moreover,  $(W_{l},b_{l})_{l = 1}^{k}$  can be chosen in such a way that all the weight matrices  $U_{l} = \mathcal{M}_{l}(W_{l})$  have full rank for every  $1\leq l\leq k$ .

Theorem 3.4 implies that a large class of CNNs employed in practice with convolutional, fully connected and max-pooling layers and standard activation functions like ReLU, sigmoid or softmax can produce linearly independent features at any hidden layer if its width is larger than the number of training samples. Figure 1 shows an example of a CNN architecture that satisfies the conditions of Theorem 3.4 at the first convolutional layer (i.e.  $k = 1$ ). Note that if a set of vectors is linearly independent then they are also linearly separable. In this sense, Theorem 3.4 suggests that deep and wide CNNs can produce linearly separable features at every wide hidden layer.

Linear separability in neural networks has been recently studied by An et al. (2015), where the authors show that a two-hidden-layer fully connected network with ReLU activation function can transform any training set to be linearly separable while approximately preserving the distances of the training data at the output layer. Compared to An et al. (2015) our Theorem 3.4 is derived for CNNs with a wider range of activation functions. Moreover, our result shows even linear independence of features which is stronger than linear separability. Recently, Nguyen & Hein (2017) have shown a similar result for deep fully connected networks and analytic activation functions.

We note that, in contrast to fully connected networks, for CNNs the condition  $n_k \geq N$  of Theorem 3.4 does not necessarily imply that the network has a huge number of parameters as the layers  $k$  and  $k + 1$  can be chosen to be convolutional. In particular, the condition  $n_k = T_k P_{k - 1} \geq N$  can be fulfilled by increasing the number of filters  $T_k$  or by using a large number of patches  $P_{k - 1}$  (however  $P_{k - 1}$  is upper bounded by  $n_k$ ), which is however only possible if  $l_{k - 1}$  is small as otherwise our condition on the patches cannot be fulfilled. Interestingly, such a architecture has been used in the VGG-Net (Simonyan & Zisserman, 2015), where they use small  $3 \times 3$  filters and minimal stride 1 in the first layer and thus they fulfill the condition  $n_k \geq N$  for  $k = 1$ , see Table 1, for ImageNet. Also note that other state-of-the-art-networks fulfill the condition in Table 1. Overall, Theorem 3.4 can be seen as a theoretical support for the usage of small filters and strides in practical CNN architectures as it increases the chance of achieving linear separability at early hidden layers in the network and also reduces the total number of training parameters. The reason why linear separability helps will be discussed in Section 4 when we analyze the loss surface of the CNNs. Note also that the condition  $n_k \geq N$  is a sufficient condition but not necessary to prove our results. In particular, we conjecture that linear separability might hold with far less number of neurons in practical applications.

One might ask now how difficult it is to find such parameters which generate linearly independent features at a hidden layer? Our next result shows that once analytic activation functions, e.g. sigmoid or softplus, are used at the first  $k$  hidden layers of the network, the linear independence of features at layer  $k$  holds with probability 1 even if one draws the parameters of the first  $k$  layers  $(W_{l},b_{l})^{k}$  randomly for any distribution which is absolutely continuous with respect to the Lebesgue measure.

Theorem 3.5 Let Assumption 3.1 hold for the training samples. Consider a deep CNN for which there exists some layer  $1 \leq k \leq L - 1$  such that

1. Every layer from 1 to  $k$  is convolutional or fully connected  
2. The width of layer  $k$  is larger than number of training samples, that is,  $n_k = T_k P_{k-1} \geq N$  
3.  $(\sigma_{1},\dots ,\sigma_{k})$  are real analytic functions and satisfy Assumption 3.2.

Then the set of parameters of the first  $k$  layers  $(W_{l},b_{l})_{l = 1}^{k}$  for which the set of feature vectors  $\{f_k(x_1),\ldots ,f_k(x_N)\}$  of layer  $k$  are not linearly independent has Lebesgue measure zero.

Note that Theorem 3.5 is a much stronger statement than Theorem 3.4, as it shows that for almost all weight configurations one gets linearly independent features at the wide layer. While Theorem 3.5 does not hold for the ReLU activation function as it is not an analytic function, we want to note again that one can approximate the ReLU function arbitrarily well using the softplus function (see Equation 6), which is an analytic function for any  $\alpha >0$  and thus Theorem 3.5 applies. It is an open question if the result holds also for the ReLU activation function itself.

It is very interesting to note that Theorem 3.5 explains previous empirical observations. In particular, Czarnecki et al. (2017) have shown empirically that linear separability is often obtained already in the first few hidden layers of the trained networks. This is done by attaching a linear classifier probe (Alain & Bengio, 2016) to every hidden layer in the network after training the whole network with backpropagation. The fact that Theorem 3.5 holds even if the parameters of the bottom layers up to the wide layer  $k$  are chosen randomly is also in line with recent empirical observations for CNN architectures that one has little loss in performance if the weights of the initial layers are chosen randomly without training (Jarrett et al., 2009; Saxe et al., 2011; Yosinski et al., 2014).

An application of Theorem 3.4 yields the following universal finite sample expressivity for CNNs. In particular, a deep CNN architecture with scalar output can perfectly express the values of any scalar-valued function over a finite number of inputs as long as the width of the last hidden layer is larger than the number of training samples.

Corollary 3.6 (Universal Finite Sample Expressivity) Let Assumption 3.1 hold for the training samples. Consider a standard CNN with scalar output which satisfies the conditions of Theorem 3.4 at the last hidden layer  $k = L - 1$ . Let  $f_{L}:\mathbb{R}^{d}\to \mathbb{R}$  be the output of the network given as

$$
f _ {L} (x) = \sum_ {j = 1} ^ {m} \lambda_ {j} f _ {(L - 1) j} (x) \quad \forall x \in \mathbb {R} ^ {d}
$$

where  $\lambda \in \mathbb{R}^{n_{L-1}}$  is the weight vector of the last layer. Then for every target output  $y \in \mathbb{R}^N$ , there exists  $\{\lambda, (W_l, b_l)_{l=1}^{L-1}\}$  so that it holds  $f_L(x_i) = y_i$  for every  $i \in [N]$ .

Proof: Since the network satisfies the conditions of Theorem 3.4 for  $k = L - 1$ , there exists a set of parameters  $(W_{l}, b_{l})_{l=1}^{L-1}$  such that  $\text{rank}(F_{L-1}) = N$ . Let  $F_{L} = [f_{L}(x_{1}), \ldots, f_{L}(x_{N})]^{T} \in \mathbb{R}^{N}$  then it follows that  $F_{L} = F_{L-1}\lambda$ . Pick  $\lambda = F_{L-1}^{T}(F_{L-1}F_{L-1}^{T})^{-1}y$  then it holds  $F_{L} = F_{L-1}\lambda = y$ .

The expressivity of neural networks has been well-studied in the literature, in particular in the universal approximation theorems for one hidden layer networks (Cybenko, 1989; Hornik et al., 1989). Recently, many results have been shown why deep networks are superior to shallow networks in terms of expressiveness (Delalleau & Bengio, 2011; Telgarsky, 2016; 2015; Eldan & Shamir, 2016; Safran & Shamir, 2017; Yarotsky, 2016; Poggio et al., 2016; Liang & Srikant, 2017; Mhaskar & Poggio, 2016; Montufar et al., 2014; Pascanu et al., 2014; Raghu et al., 2017). While most of these results are derived for fully connected networks, it seems that Cohen & Shashua (2016) are the first ones who study expressivity of CNNs. In particular, they show that CNNs with max-pooling and ReLU units are universal in the sense that they can approximate any given function if the size of the networks is unlimited. However, the number of convolutional filters in this result has to grow exponentially with the number of patches and they do not allow shared weights in their result, which is a standard feature of CNNs. Our Corollary 3.6 shows universal finite sample expressivity, instead of universal function approximation, even for  $L = 2$  and  $k = 1$ , that is a single convolutional layer network can perfectly fit the training data as long as the number of hidden units is not smaller than the number of samples. To the best of our knowledge, this is the first result on universal finite sample expressivity for a large class of practical CNNs.

For fully connected networks, universal finite sample expressivity has been studied by Zhang et al. (2017); Nguyen & Hein (2017); Hardt & Ma (2017). They show that a network with a single hidden layer with  $N$  hidden units can express any training set of size  $N$ . While the number of training parameters of a single hidden layer CNN with  $N$  hidden units and scalar output is just  $2N + T_1l_0$ , where  $T_{1}$  is the number of convolutional filters and  $l_{0}$  is the length of each filter, it is  $Nd + 2N$  for fully connected networks. If we set the width of the hidden layer of the CNN as  $n_1 = T_1P_0 = N$

![](images/3f661bd0d735fc4269aaf4f269a5dbbafb06bf08ab0de83c7a1238b6499b2a03.jpg)  
Figure 1: An example of CNN for a given training set of size  $N \leq 100 \times 26 \times 26 = 67600$ . The width of each layer is  $d = n_0 = 784$ ,  $n_1 = 67600$ ,  $n_2 = 16900$ ,  $n_3 = 2880$ ,  $n_4 = 720$ ,  $n_5 = 100$ ,  $n_6 = m = 10$ . One can see that  $n_1 \geq N$  and the network has pyramidal structure from layer 2 till the output layer, that is,  $n_2 \geq \ldots \geq n_6$ .

Table 1: The width of the first convolutional layer  $(n_{1})$  and the maximum width of all the hidden layers  $(\max_{1\leq k\leq L - 1}n_k)$  of state-of-the-art CNN architectures in comparison with the size of ImageNet  $(N\approx 1200K)$ . All numbers are lower bounds on the true width.  

<table><tr><td>CNN Architecture</td><td>n1</td><td>maxknk</td></tr><tr><td>VGG(A-E) (Simonyan &amp; Zisserman, 2015)</td><td>3000K</td><td>3000K</td></tr><tr><td>InceptionV3 (Szegedy et al., 2015b)</td><td>700K</td><td>1300K</td></tr><tr><td>InceptionV4 (Szegedy et al., 2016)</td><td>700K</td><td>1300K</td></tr><tr><td>SqueezeNet (Iandola et al., 2016)</td><td>1180K</td><td>1180K</td></tr><tr><td>Enet (Paszke et al., 2016)</td><td>1000K</td><td>1000K</td></tr><tr><td>GoogLeNet (Szegedy et al., 2015a)</td><td>800K</td><td>800K</td></tr><tr><td>ResNet (He et al., 2016)</td><td>800K</td><td>800K</td></tr><tr><td>Xception (Chollet, 2016)</td><td>700K</td><td>700K</td></tr></table>

in order to fulfill the condition of Corollary 3.6, then the number of training parameters of the CNN becomes  $2N + Nl_0 / P_0$ , which is less than  $3N$  if  $l_0 \leq P_0$  compared to  $(d + 2)N$  for the fully connected case. In practice one almost always has  $l_0 \leq P_0$  as  $l_0$  is typically a small integer and  $P_0$  is on the order of the dimension of the input. Thus, the number of parameters of the CNN to achieve universal finite sample expressivity is significantly smaller than that of fully connected networks.

Obviously, in practice it is more important that the network generalizes rather than just fitting the training data. By using shared weights and sparsity structure, CNNs seem to implicitly regularize the model class in order to achieve good generalization performance. Thus even though they can fit also random labels or noise (Zhang et al., 2017) due to the universal finite sample expressivity shown in Corollary 3.6, they seem still to be able to generalize well (Zhang et al., 2017).

# 4 THE LOSS SURFACE OF CONVOLUTIONAL NEURAL NETWORKS

In this section, we restrict our analysis to the use of least squares loss. However, as we show later that the network can produce exactly the target output (i.e.  $F_{L} = Y$ ) for some choice of parameters, all our results can also be extended to any other loss function where the global minimum is attained at  $F_{L} = Y$ , for instance the squared Hinge-loss analyzed in Nguyen & Hein (2017). Let  $\mathcal{P}$  denote the space of all parameters of the network. The final training objective  $\Phi : \mathcal{P} \to \mathbb{R}$  is given as

$$
\Phi \left(\left(W _ {l}, b _ {l}\right) _ {l = 1} ^ {L}\right) = \frac {1}{2} \| F _ {L} - Y \| _ {F} ^ {2} \tag {7}
$$

where  $F_{L}$  is defined as in (5), which is also the same as

$$
F _ {L} = \sigma_ {L - 1} (\dots \sigma_ {1} (X U _ {1} + b _ {1}) \dots) U _ {L} + b _ {L},
$$

where  $U_{l} = \mathcal{M}_{l}(W_{l})$  for every  $1\leq l\leq L$

Our assumptions on the architecture of CNNs is given below.

Assumption 4.1 (CNN Architecture) Every layer in the network is a convolutional layer or fully connected layer and the output layer is fully connected. Moreover, there exists some hidden layer  $1 \leq k \leq L - 1$  such that the following holds:

- The width of layer  $k$  is larger than number of training samples, that is,  $n_k = T_k P_{k-1} \geq N$  
- All the activation functions of the hidden layers  $(\sigma_{1},\dots,\sigma_{L - 1})$  satisfy Assumption 3.2  
-  $(\sigma_{k+1}, \ldots, \sigma_{L-1})$  are strictly increasing or strictly decreasing, and differentiable  
- The network is pyramidal from layer  $k + 1$  till the output layer, that is,  $n_{k + 1} \geq \ldots \geq n_L$

A typical example that satisfies Assumption 4.1 is the following (see Figure 1 for an illustration):

- The first layer is a convolutional layer with  $n_1 = T_1P_0 \geq N$  
- Every layer from layer 2 till the output layer is convolutional or fully connected  
-  $(\sigma_{1},\ldots ,\sigma_{k})$  can be ReLU, sigmoid or softplus  
-  $(\sigma_{k+1}, \ldots, \sigma_{L-1})$  can be sigmoid or softplus  
-  $n_2 \geq n_3 \geq \ldots \geq n_L$

One can easily check that the above example satisfies Assumption 4.1 for  $k = 1$ .

In the following, let us define for every  $1 \leq k \leq L - 1$  the subset  $S_{k} \subseteq \mathcal{P}$  such that

$$
S _ {k} := \left\{\left(W _ {l}, b _ {l}\right) _ {l = 1} ^ {L} \mid r a n k (F _ {k}) = N \text {a n d} U _ {l} \text {h a s f u l l r a n k f o r e v e r y} l \in [ k + 2, L ] \right\}.
$$

The set  $S_{k}$  is the set of parameters where the feature vectors at layer  $k$  are linearly independent and all the weight matrices from layer  $k + 2$  till the output layer have full rank. In the following, we examine conditions for global optimality in  $S_{k}$ . It is important to note that  $S_{k}$  covers almost the whole parameter space under an additional mild condition on the activation function.

Lemma 4.2 Let Assumption 3.1 hold for the training sample and let the CNN architecture satisfy Assumption 4.1 for some layer  $1 \leq k \leq L - 1$ . We assume further that the activation functions of the first  $k$  layers  $(\sigma_1, \ldots, \sigma_k)$  are real analytic. Then the set  $\mathcal{P} \setminus S_k$  has Lebesgue measure zero.

Proof: One can see that

$$
\mathcal {P} \setminus S _ {k} \subseteq \left\{\left(W _ {l}, b _ {l}\right) _ {l = 1} ^ {L} \mid r a n k (F _ {k}) <   N \right\} \cup \left\{\left(W _ {l}, b _ {l}\right) _ {l = 1} ^ {L} \mid U _ {l} \text {h a s l o w r a n k f o r s o m e l a y e r} l \right\}.
$$

By Theorem 3.5, it holds that the set  $\{(W_l, b_l)_{l=1}^L \mid \text{rank}(F_k) < N\}$  has Lebesgue measure zero. Moreover, it follows from Lemma 2.5 that the set  $\{(W_l, b_l)_{l=1}^L \mid U_l$  has low rank for some layer  $l\}$  also has measure zero. Thus,  $\mathcal{P} \setminus S_k$  has Lebesgue measure zero.

In the next key lemma, we bound the objective function in terms of its gradient magnitude w.r.t. the weight matrix of layer  $k$  for which  $n_k \geq N$ . For every matrix  $A \in \mathbb{R}^{m \times n}$ , let  $\sigma_{\min}(A)$  and  $\sigma_{\max}(A)$  denote the smallest and largest singular value of  $A$ . Let  $\| A \|_F = \sqrt{\sum_{i,j} A_{ij}^2}$ ,  $\| A \|_{\min} \coloneqq \min_{i,j} |A_{ij}|$  and  $\| A \|_{\max} \coloneqq \max_{i,j} |A_{ij}|$ . From Equations (4), (5) and (7), it follows that  $\Phi$  can be seen as a function of  $(U_l, b_l)_{l=1}^L$ , and thus we can use  $\nabla_{U_k} \Phi$ . If layer  $k$  is fully connected then  $U_k = \mathcal{M}_k(W_k) = W_k$  and thus  $\nabla_{U_k} \Phi = \nabla_{W_k} \Phi$ . Otherwise, if layer  $k$  is convolutional then we note that  $\nabla_{U_k} \Phi$  is "not" the true gradient of the training objective because  $U_k$  is not the true optimization parameter but  $W_k$ . In this case, the true gradient of  $\Phi$  w.r.t. to the true parameter matrix  $W_k$  which consists of convolutional filters can be computed via chain rule as

$$
\frac {\partial \Phi}{\partial (W _ {k}) _ {r s}} = \sum_ {i, j} \frac {\partial \Phi}{\partial (U _ {k}) _ {i j}} \frac {\partial (U _ {k}) _ {i j}}{\partial (W _ {k}) _ {r s}}
$$

Please note that even though we write the partial derivatives with respect to the matrix elements,  $\nabla_{W_k}\Phi$  resp.  $\nabla_{U_k}\Phi$  are the matrices of the same dimension as  $W_{k}$  resp.  $U_{k}$  in the following.

Lemma 4.3 Consider a standard deep CNN which satisfies Assumption 4.1 for some hidden layer  $1 \leq k \leq L - 1$ . Then it holds

$$
\left\| \nabla_ {U _ {k + 1}} \Phi \right\| _ {F} \geq \sigma_ {\min } (F _ {k}) \Big (\prod_ {l = k + 1} ^ {L - 1} \sigma_ {\min } (U _ {l + 1}) \| \sigma_ {l} ^ {\prime} (G _ {l}) \| _ {\min } \Big) \| F _ {L} - Y \| _ {F}
$$

and

$$
\left\| \nabla_ {U _ {k + 1}} \Phi \right\| _ {F} \leq \sigma_ {\max } (F _ {k}) \Big (\prod_ {l = k + 1} ^ {L - 1} \sigma_ {\max } (U _ {l + 1}) \| \sigma_ {l} ^ {\prime} (G _ {l}) \| _ {\max } \Big) \| F _ {L} - Y \| _ {F}.
$$

Our next main result is motivated by the fact that empirically when training over-parameterized neural networks with shared weights and sparsity structure like CNNs, there seem to be no problems with sub-optimal local minima. In many cases, even when training labels are completely random, local search algorithms like stochastic gradient descent can converge to a solution with almost zero training error (Zhang et al., 2017). To understand better this phenomenon, we first characterize in the following Theorem 4.4 the set of points in parameter space with zero loss, and then analyze in Theorem 4.5 the loss surface for a special case of the network. We emphasize that our results hold for standard deep CNNs with convolutional layers with shared weights and fully connected layers.

Theorem 4.4 (Necessary and Sufficient Condition for Zero Training Error) Let Assumption 3.1 hold for the training sample and suppose that the CNN architecture satisfies Assumption 4.1 for some hidden layer  $1 \leq k \leq L - 1$ . Let  $\Phi : \mathcal{P} \to \mathbb{R}$  be defined as in (7). Given any point  $(W_{l}, b_{l})_{l = 1}^{L} \in S_{k}$ , then it holds that  $\Phi \left((W_{l}, b_{l})_{l = 1}^{L}\right) = 0$  if and only if  $\nabla_{U_{k + 1}}\Phi|_{(W_{l}, b_{l})_{l = 1}^{L}} = 0$ .

Proof: If  $\Phi\left((W_l, b_l)_{l=1}^L\right) = 0$  then it follows from the upper bound of Lemma 4.3 that  $\nabla_{U_{k+1}} \Phi = 0$ . Now, we suppose that  $\nabla_{U_{k+1}} \Phi = 0$ . Since  $(W_l, b_l)_{l=1}^L \in S_k$  it holds  $\text{rank}(F_k) = N$  and  $U_l$  has full rank for every  $l \in [k+2,L]$ . Thus it holds  $\sigma_{\min}(F_k) > 0$  and  $\sigma_{\min}(U_l) > 0$  for every  $l \in [k+2,L]$ . Moreover,  $(\sigma_{k+1}, \ldots, \sigma_{L-1})$  have non-zero derivative by Assumption 4.1 and thus  $\| \sigma_l'(G_l) \|_{\min} > 0$  for every  $l \in [k+1,L-1]$ . This combined with the lower bound in Lemma 4.3 leads to  $\| F_L - Y \|_F = 0$  and thus  $\Phi\left(W_l, b_l\right)_{l=1}^L = 0$ .

Lemma 4.2 shows that the set of points which are not covered by Theorem 4.4 has just measure zero under a mild condition. The necessary and sufficient condition of Theorem 4.4 is rather intuitive as it requires the gradient of the training objective to vanish w.r.t. the full weight matrix of layer  $k + 1$  regardless of the architecture of this layer. It turns out that if layer  $k + 1$  is fully connected, then this condition is always satisfied at a critical point, in which case we obtain that every critical point in  $S_{k}$  is a global minimum with exact zero training error. This is shown in the next Theorem 4.5, where we consider a classification task with  $m$  classes.  $Z \in \mathbb{R}^{m \times m}$  is the full rank class encoding matrix e.g. the identity matrix and  $(X,Y)$  the training sample such that  $Y_{i} = Z_{j}$ : whenever  $x_{i}$  belongs to class  $j$  for every  $i \in [N], j \in [m]$ .

Theorem 4.5 (Loss Surface of CNNs) Let  $(X,Y,Z)$  be a training set for which Assumption 3.1 holds, the CNN architecture satisfies Assumption 4.1 for some hidden layer  $1\leq k\leq L - 1$  , and layer  $k + 1$  is fully connected. Let  $\Phi :\mathcal{P}\to \mathbb{R}$  be defined as in (7). Then

- Every critical point  $(W_{l},b_{l})_{l = 1}^{L}\in S_{k}$  is a global minimum with  $\Phi \Big((W_l,b_l)_l = 1\Big) = 0$  
- There exist infinitely many global minima  $(W_{l},b_{l})_{l = 1}^{L}\in S_{k}$  with  $\Phi \Big((W_l,b_l)_l = 1\Big) = 0$

Theorem 4.5 indicates that the loss surface for this type of CNNs has a rather simple structure in the sense that almost every critical point is a global minimum with zero training error. It remains an interesting open problem if this result can be transferred to the case where layer  $k + 1$  is also convolutional. In any case whether layer  $k + 1$  is fully connected or not, one might still assume that a solution with zero training error still exists. However, note that Theorem 4.4 shows that at those points where the loss is zero, the gradient of  $\Phi$  w.r.t.  $U_{k + 1}$  must be zero as well. An interesting special case of Theorem 4.5 is when the network is fully connected, in which case all the results of Theorem 4.5 hold without any modifications.

Corollary 4.6 (Loss Surface of Fully Connected Nets) Let  $(X,Y,Z)$  be a training set with non-identical training samples, i.e.  $x_{i}\neq x_{j}$  for every  $i\neq j$  and the fully connected network satisfies Assumption 4.1 for some layer  $1\le k\le L - 1$ . Let  $\Phi :\mathcal{P}\to \mathbb{R}$  be defined as in (7). Then the following holds

- Every critical point  $(W_{l},b_{l})_{l = 1}^{L}\in S_{k}$  is a global minimum with  $\Phi \Big((W_l,b_l)_l = 1\Big) = 0$  
- There exist infinitely many global minima  $(W_{l},b_{l})_{l = 1}^{L}\in S_{k}$  with  $\Phi \Big((W_l,b_l)_{l = 1}^L\Big) = 0$

Corollary 4.6 can be seen as a formal proof for the implicit assumption used in the recent work (Nguyen & Hein, 2017) that there exists a global minimum with zero training error for the class of fully connected, deep and wide networks.

# 5 CONCLUSION

We have analyzed the expressiveness and loss surface of CNNs in realistic and practically relevant settings. As state-of-the-art networks fulfill exactly or approximately the condition to have a sufficiently wide convolutional layer, we think that our results help to understand why current CNNs can be trained so effectively. It would be interesting to discuss the loss surface for cross-entropy loss, which currently does not fit into our analysis as the global minimum does not exist when the data is linearly separable.

# REFERENCES

G. Alain and Y. Bengio. Understanding intermediate layers using linear classifier probes. In ICLR Workshop, 2016.  
S. An, F. Boussaid, and M. Bennamoun. How can deep rectifier networks achieve linear separability and preserve distances? In ICML, 2015.  
A. Andoni, R. Panigrahy, G. Valiant, and L. Zhang. Learning polynomials with neural networks. In ICML, 2014.  
P. Auer, M. Herbster, and M. K. Warmuth. Exponentially many local minima for single neurons. In NIPS, 1996.  
P. Baldi and K. Hornik. Neural networks and principle component analysis: Learning from examples without local minima. *Neural Networks*, 2:53-58, 1988.  
A. Blum and R. L Rivest. Training a 3-node neural network is np-complete. In NIPS, 1989.  
A. Brutzkus and A. Globerson. Globally optimal gradient descent for a convnet with gaussian inputs, 2017. arXiv:1702.07966.  
F. Chollet. Xception: Deep learning with depthwise separable convolutions, 2016. arXiv:1610.02357.  
A. Choromanska, M. Hena, M. Mathieu, G. B. Arous, and Y. LeCun. The loss surfaces of multilayer networks. In AISTATS, 2015a.  
A. Choromanska, Y. LeCun, and G. B. Arous. Open problem: The landscape of the loss surfaces of multilayer networks.  $COLT$ , 2015b.  
N. Cohen and A. Shashua. Convolutional rectifier networks as generalized tensor decompositions. In ICML, 2016.  
G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems, 2:303-314, 1989.  
W. M. Czarnecki, G. Swirszcz, M. Jaderberg, S. Osindero, O. Vinyals, and K. Kavukcuoglu. Understanding synthetic gradients and decoupled neural interfaces. In ICML, 2017.

Y. Dauphin, R. Pascanu, C. Gulcehre, K. Cho, S. Ganguli, and Y. Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. In NIPS, 2014.  
O. Delalleau and Y. Bengio. Shallow vs. deep sum-product networks. In NIPS, 2011.  
S. S. Du, J. D. Lee, and Y. Tian. When is a convolutional filter easy to learn?, 2017. arXiv:1709.06129.  
R. Eldan and O. Shamir. The power of depth for feedforward neural networks. In  $COLT$ , 2016.  
C. D. Freeman and J. Bruna. Topology and geometry of half-rectified network optimization. In ICLR, 2017.  
A. Gautier, Q. Nguyen, and M. Hein. Globally optimal training of generalized polynomial neural networks with nonlinear spectral methods. In NIPS, 2016.  
S. Goel and A. Klivans. Learning depth-three neural networks in polynomial time, 2017. arXiv:1709.06010.  
I. J. Goodfellow, O. Vinyals, and A. M. Saxe. Qualitatively characterizing neural network optimization problems. In ICLR, 2015.  
B. D. Haeffele and R. Vidal. Global optimality in tensor factorization, deep learning, and beyond, 2015. arXiv:1506.07540v1.  
M. Hardt and T. Ma. Identity matters in deep learning. In ICLR, 2017.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.  
K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural Networks, 2:359-366, 1989.  
F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer. Squeezezenet: Alexnet-level accuracy with 50x fewer parameters and  $< 0.5\mathrm{mb}$  model size, 2016. arXiv:1602.07360.  
M. Janzamin, H. Sedghi, and A. Anandkumar. Beating the perils of non-convexity: Guaranteed training of neural networks using tensor methods. arXiv:1506.08473, 2016.  
K. Jarrett, K. Kavukcuoglu, and Y. LeCun. What is the best multi-stage architecture for object recognition? In CVPR, 2009.  
K. Kawaguchi. Deep learning without poor local minima. In NIPS, 2016.  
S. G. Krantz and H. R. Parks. A Primer of Real Analytic Functions. Birkhäuser, Boston, second edition, 2002.  
A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.  
Y. LeCun, B. Boser, J.S. Denker, D. Henderson, R.E. Howard, W. Hubbard, and L.D. Jackel. Handwritten digit recognition with a back-propagation network. In NIPS, 1990.  
Y. Li and Y. Yuan. Convergence analysis of two-layer neural networks with relu activation, 2017. arXiv:1705.09886.  
S. Liang and R. Srikant. Why deep neural networks for function approximation? In ICLR, 2017.  
R. Livni, S. Shalev-Shwartz, and O. Shamir. On the computational efficiency of training neural networks. In NIPS, 2014.  
A. Mahendran and A. Vedaldi. Understanding deep image representations by inverting them. In CVPR, 2015.  
H. Mhaskar and T. Poggio. Deep vs. shallow networks: An approximation theory perspective, 2016. arXiv:1608.03287.

B. Mityagin. The zero set of a real analytic function, 2015. arXiv:1512.07276.  
G. Montufar, R. Pascanu, K. Cho, and Y. Bengio. On the number of linear regions of deep neural networks. In NIPS, 2014.  
Q. Nguyen and M. Hein. The loss surface of deep and wide neural networks. In ICML, 2017.  
V. D. Nguyen. Complex powers of analytic functions and meromorphic renormalization in qft, 2015. arXiv:1503.00995.  
R. Pascanu, G. Montufar, and Y. Bengio. On the number of response regions of deep feedforward networks with piecewise linear activations. In ICLR, 2014.  
A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello. Enet: A deep neural network architecture for real-time semantic segmentation, 2016. arXiv:1606.02147.  
T. Poggio, H. Mhaskar, L. Rosasco, B. Miranda, and Q. Liao. Why and when can deep - but not shallow - networks avoid the curse of dimensionality: a review, 2016. arXiv:1611.00740.  
M. Raghu, B. Poole, J. Kleinberg, S. Ganguli, and J. Sohl-Dickstein. On the expressive power of deep neural networks. In ICML, 2017.  
I. Safran and O. Shamir. On the quality of the initial basin in overspecified networks. In ICML, 2016.  
I. Safran and O. Shamir. Depth-width tradeoffs in approximating natural functions with neural networks. In ICML, 2017.  
A. Saxe, P. W. Koh, Z. Chen, M. Bhand, B. Suresh, and A. Y. Ng. On random weights and unsupervised feature learning. In ICML, 2011.  
H. Sedghi and A. Anandkumar. Provable methods for training neural networks with sparse connectivity. In ICLR Workshop, 2015.  
S. Shalev-Shwartz, O. Shamir, and S. Shammah. Failures of gradient-based deep learning. In ICML, 2017.  
O. Shamir. Distribution-specific hardness of learning neural networks, 2017. arXiv:1609.01037.  
J. Sima. Training a single sigmoidal neuron is hard. Neural Computation, 14:2709-2728, 2002.  
K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.  
M. Soltanolkotabi. Learning relus via gradient descent, 2017. arXiv:1705.04591.  
D. Soudry and E. Hoffer. Exponentially vanishing sub-optimal local minima in multilayer neural networks, 2017. arXiv:1702.05777.  
C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015a.  
C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision, 2015b. arXiv:1512.00567.  
C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning, 2016. arXiv:1602.07261.  
M. Telgarsky. Representation benefits of deep feedforward networks, 2015. arXiv:1509.08101v2.  
M. Telgarsky. Benefits of depth in neural networks. In  $COLT$ , 2016.  
Y. Tian. An analytical formula of population gradient for two-layered relu network and its applications in convergence and critical point analysis. In ICML, 2017.  
D. Yarotsky. Error bounds for approximations with deep relu networks, 2016. arXiv:1610.01145.

J. Yosinski, J. Clune, Y. Bengio, and H. Lipson. How transferable are features in deep neural networks? In NIPS, 2014.  
J. Yosinski, J. Clune, A. Nguyen, T. Fuchs, and H. Lipson. Understanding neural networks through deep visualization. In ICML, 2015.  
C. Yun, S. Sra, and A. Jabbabaie. Global optimality conditions for deep neural networks, 2017. arXiv:1707.02444.  
M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In ECCV, 2014.  
C. Zhang, S. Bengio, M. Hardt, B. Recht, and Oriol Vinyals. Understanding deep learning requires re-thinking generalization. In ICLR, 2017.  
K. Zhong, Z. Song, P. Jain, P. Bartlett, and I. Dhillon. Recovery guarantees for one-hidden-layer neural networks. In ICML, 2017.
