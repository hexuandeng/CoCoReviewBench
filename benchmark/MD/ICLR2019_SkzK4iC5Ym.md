# DIMINISHING BATCH NORMALIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this paper, we propose a generalization of the BN algorithm, diminishing batch normalization (DBN), where we update the BN parameters in a diminishing moving average way. Batch normalization (BN) is very effective in accelerating the convergence of a neural network training phase that it has become a common practice. Our proposed DBN algorithm remains the overall structure of the original BN algorithm while introduces a weighted averaging update to some trainable parameters. We provide an analysis of the convergence of the DBN algorithm that converges to a stationary point with respect to trainable parameters. Our analysis can be easily generalized for original BN algorithm by setting some parameters to constant. To the best knowledge of authors, this analysis is the first of its kind for convergence with Batch Normalization introduced. We analyze a two-layer model with arbitrary activation function. The primary challenge of the analysis is the fact that some parameters are updated by gradient while others are not. The convergence analysis applies to any activation function that satisfies our common assumptions. For the analysis, we also show the sufficient and necessary conditions for the stepsizes and diminishing weights to ensure the convergence. In the numerical experiments, we use more complex models with more layers and ReLU activation. We observe that DBN outperforms the original BN algorithm on Imagenet, MNIST, NI and CIFAR-10 datasets with reasonable complex FNN and CNN models.

# 1 INTRODUCTION

Deep neural networks (DNN) have shown unprecedented success in various applications such as object detection. However, it still takes a long time to train a DNN until it converges. Ioffe & Szegedy identified a critical problem involved in training deep networks, internal covariate shift, and then proposed batch normalization (BN) to decrease this phenomenon. BN addresses this problem by normalizing the distribution of every hidden layer's input. In order to do so, it calculates the preactivation mean and standard deviation using mini-batch statistics at each iteration of training and uses these estimates to normalize the input to the next layer. The output of a layer is normalized by using the batch statistics, and two new trainable parameters per neuron are introduced that capture the inverse operation. It is now a standard practice Bottou et al. (2016); He et al. (2016). While this approach leads to a significant performance jump, to the best of our knowledge, there is no known theoretical guarantee for the convergence of an algorithm with BN. The difficulty of analyzing the convergence of the BN algorithm comes from the fact that not all of the BN parameters are updated by gradients. Thus, it invalidates most of the classical studies of convergence for gradient methods.

In this paper, we propose a generalization of the BN algorithm, diminishing batch normalization (DBN), where we update the BN parameters in a diminishing moving average way. It essentially means that the BN layer adjusts its output according to all past mini-batches instead of only the current one. It helps to reduce the problem of the original BN that the output of a BN layer on a particular training pattern depends on the other patterns in the current mini-batch, which is pointed out by Bottou et al.. By setting the layer parameter we introduce into DBN to a specific value, we recover the original BN algorithm.

We give a convergence analysis of the algorithm with a two-layer batch-normalized neural network and diminishing step sizes. We assume two layers (the generalization to multiple layers can be made by using the same approach but substantially complicating the notation) and an arbitrary loss function. The convergence analysis applies to any activation function that follows our common

assumption. The main result shows that under diminishing step sizes on gradient updates and updates on mini-batch statistics, and standard Lipschitz conditions on loss functions DBN converges to a stationary point. As already pointed out the primary challenge is the fact that some trainable parameters are updated by gradient while others are updated by a minor recalculation.

Contributions. The main contribution of this paper is in providing a general convergence guarantee for DBN. Specifically, we make the following contributions.

- In section 4, we show the sufficient and necessary conditions for the stepsizes and diminishing weights to ensure the convergence of BN parameters.  
- We show that the algorithm converges to a stationary point under a general nonconvex objective function.

This paper is organized as follows. In Section 2, we review the related works and the development of the BN algorithm. We formally state our model and algorithm in Section 3. We present our main results in Sections 4. In Section 5, we numerically show that the DBN algorithm outperforms the original BN algorithm. Proofs for main steps are collected in the Appendix.

# 2 LITERATURE REVIEW

Before the introduction of BN, it has long been known in the deep learning community that input whitening and decorrelation help to speed up the training process. In fact, Orr & Müller show that preprocessing the data by subtracting the mean, normalizing the variance, and decorrelating the input has various beneficial effects for back-propagation. Krizhevsky et al. propose a method called local response normalization which is inspired by computational neuroscience and acts as a form of lateral inhibition, i.e., the capacity of an excited neuron to reduce the activity of its neighbors. Gülçehre & Bengio propose a standardization layer that bears significant resemblance to batch normalization, except that the two methods are motivated by very different goals and perform different tasks.

Inspired by BN, several new works are taking BN as a basis for further improvements. Layer normalization Ba et al. (2016) is much like the BN except that it uses all of the summed inputs to compute the mean and variance instead of the mini-batch statistics. Besides, unlike BN, layer normalization performs precisely the same computation at training and test times. Normalization propagation that Arpit et al. uses data-independent estimations for the mean and standard deviation in every layer to reduce the internal covariate shift and make the estimation more accurate for the validation phase. Weight normalization also removes the dependencies between the examples in a minibatch so that it can be applied to recurrent models, reinforcement learning or generative models Salimans & Kingma (2016). Cooijmans et al. propose a new way to apply batch normalization to RNN and LSTM models.

Given all these flavors, the original BN method is the most popular technique and for this reason our choice of the analysis. To the best of our knowledge, we are not aware of any prior analysis of BN.

BN has the gradient and non-gradient updates. Thus, nonconvex convergence results do not immediately transfer. Our analysis explicitly considers the workings of BN. However, nonconvex convergence proofs are relevant since some small portions of our analysis rely on known proofs and approaches.

Neural nets are not convex, even if the loss function is convex. For classical convergence results with a nonconvex objective function and diminishing learning rate, we refer to survey papers Bertsekas (2011); Bertsekas & Tsitsiklis (2000); Bottou et al. (2016). Bertsekas & Tsitsiklis provide a convergence result with the deterministic gradient with errors. Bottou et al. provide a convergence result with the stochastic gradient. The classic analyses showing the norm of gradients of the objective function going to zero date back to Grippo (1994); Polyak & Tsypkin (1973); Polyak (1987). For strongly convex objective functions with a diminishing learning rate, we learn the classic convergence results from Bottou et al..

# 3 MODEL AND ALGORITHM

The optimization problem for a network is an objective function consisting of a large number of component functions, that reads:

$$
\min  \bar {f} (\theta , \lambda) = \sum_ {i = 1} ^ {N} f _ {i} \left(X _ {i}: \theta , \lambda\right), \tag {1}
$$

subject to  $\theta \in P, \lambda \in Q$

where  $f_{i}:\mathbb{R}^{n_{1}}\times \mathbb{R}^{n_{2}}\to \mathbb{R}, i = 1,\dots,N,$  are real-valued functions for any data record  $X_{i}$ . Index  $i$  associates with data record  $X_{i}$  and target response  $y_{i}$  (hidden behind the dependency of  $f$  on  $i$ ) in the training set. Parameters  $\theta$  include the common parameters updated by gradients directly associated with the loss function, i.e., behind the part that we have a parametric model, while BN parameters  $\lambda$  are introduced by the BN algorithm and not updated by gradient methods but by the mini-batch statistics. We define that the derivative of  $f_{i}$  is always taken with respect to  $\theta$ :

$$
\nabla f _ {i} \left(X _ {i}: \theta , \lambda\right) := \nabla_ {\theta} f _ {i} \left(X _ {i}: \theta , \lambda\right). \tag {2}
$$

The deep network we analyze has 2 fully-connected layers with  $D_{1}$  neurons each. The techniques presented can be extended to more layers with additional notation. Each hidden layer computes  $y = a(Wu)$  with activation function  $a(\cdot)$  and  $u$  is the input vector of the layer. We do not need to include an intercept term since the BN algorithm automatically adjusts for it. BN is applied to the output of the first hidden layer.

![](images/bbd819d75ae4c4f903df4b7c5e8ad0e40c631142601162201e94be87245b9292.jpg)  
Figure 1: The structure of our batch-normalized network model in the analysis.

We next describe the computation in each layer to show how we obtain the output of the network. The notations introduced here is used in the analysis. Figure 1 shows the full structure of the network. The input data is vector  $X$ , which is one of  $\{X_{i}\}_{i = 1}^{N}$ . Vector  $\lambda = ((\mu_j)_{j = 1}^D, (\sigma_j)_{j = 1}^D)$  is the set of all BN parameters and vector  $\theta = (W_1, W_2, (\beta_j^{(1)})_{j = 1}^D, (\gamma_j^{(1)})_{j = 1}^D)$  is the set of all trainable parameters which are updated by gradients.

Matrices  $W_{1}, W_{2}$  are the actual model parameters and  $\beta, \gamma$  are introduced by BN. The value of  $j^{th}$  neuron of the first hidden layer is

$$
z _ {j} ^ {(1)} (X: \theta) = a \left(W _ {1, j, \cdot} X\right), \tag {3}
$$

where  $W_{1,j}$  denotes the weights of the linear transformations for the  $j^{th}$  neuron.

The  $j^{th}$  entry of batch-normalized output of the first layer is

$$
y _ {j} ^ {(1)} (X: \theta , \lambda) = \gamma_ {j} ^ {(1)} \left(\frac {z _ {j} ^ {(1)} (X : \theta) - \mu_ {j}}{\sigma_ {j} + \epsilon_ {B}}\right) + \beta_ {j} ^ {(1)},
$$

where  $\beta_{j}^{(1)}$  and  $\gamma_{j}^{(1)}$  are trainable parameters updated by gradient and  $\mu_{j}$  and  $\sigma_{j}$  are batch normalization parameters for  $z_{j}^{(1)}$ . Trainable parameter  $\mu_{j}$  is the mini-batch mean of  $z_{j}^{(1)}$  and trainable parameter  $\sigma_{j}$  is the mini-batch sample deviation of  $z_{j}^{(1)}$ . Constant  $\epsilon_{B}$  keeps the denominator from zero. The output of  $j^{th}$  entry of the output layer is:

$$
z _ {j} ^ {(2)} (X: \theta) = a \left(W _ {2, j, \cdot} \left[ \gamma_ {j} ^ {(1)} \left(\frac {z _ {j} ^ {(1)} (X : \theta) - \mu_ {j}}{\sigma_ {j} + \epsilon_ {B}}\right) + \beta_ {j} ^ {(1)} \right]\right) \tag {4}
$$

The objective function for the  $i^{th}$  sample is

$$
f _ {i} \left(X _ {i}: \theta , \lambda\right) = l _ {i} \left(\left(z _ {j} ^ {(2)} \left(X _ {i}: \theta , \lambda\right)\right) _ {j}\right), \tag {5}
$$

where  $l_{i}(\cdot)$  is the loss function associated with the target response  $y_{i}$ . For sample  $i$ , we have the following complete expression for the objective function:

$$
f _ {i} \left(X _ {i}: \theta , \lambda\right) = l _ {i} \left(a \left(\sum_ {j = 1} ^ {D} W _ {2, k, j} \left[ \gamma_ {j} ^ {(1)} \frac {a \left(W _ {1 , j} , X _ {i} - \mu_ {j}\right)}{\sigma_ {j} + \epsilon_ {B}} + \beta_ {j} ^ {(1)} \right]\right) _ {k}\right). \tag {6}
$$

Function  $f_{i}(X_{i}:\theta ,\lambda)$  is nonconvex with respect to  $\theta$  and  $\lambda$ .

# 3.1 ALGORITHM

Algorithm 1 shows the algorithm studied herein. There are two deviations from the standard BN algorithm, one of them actually being a generalization. We use the full gradient instead of the more popular stochastic gradient (SG) method. It essentially means that each batch contains the entire training set instead of a randomly chosen subset of the training set. An analysis of SG is potential future research. Although the primary motivation for full gradient update is to reduce the burdensome in showing the convergence, the full gradient method is similar to SG in the sense that both of them go through the entire training set, while full gradient goes through it deterministically and the SG goes through it in expectation. Therefore, it is reasonable to speculate that the SG method has similar convergence property as the full algorithm studied herein.

Algorithm 1 DBN: Diminishing Batch-Normalized Network Update Algorithm  
1: Initialize  $\theta \in \mathbb{R}^{n_1}$  and  $\lambda \in \mathbb{R}^{n_2}$   
2: for iteration  $m = 1,2,\ldots$  do  
3:  $\theta^{(m + 1)}\coloneqq \theta^{(m)} - \eta^{(m)}\sum_{i = 1}^{N}\nabla f_i(X_i:\theta^{(m)},\lambda^{(m)})$   
4: for  $j = 1,\dots,D_1$  do  
5:  $\mu_j^{(m + 1)}\coloneqq \frac{1}{N}\sum_{i = 1}^N z_j^{(1)}(X_i:\theta^{(m + 1)})$   
6:  $\sigma_j^{(m + 1)}\coloneqq \sqrt{\frac{1}{N}\sum_{i = 1}^N\left(z_j^{(1)}(X_i:\theta^{(m + 1)}) - \mu_j^{(m + 1)}\right)^2}$   
7:  $\lambda^{(m + 1)}\coloneqq \alpha^{(m + 1)}\left((\mu_j^{(m + 1)})_{j = 1}^{D_1},(\sigma_j^{(m + 1)})_{j = 1}^{D_1}\right) + (1 - \alpha^{(m + 1)})\lambda^{(m)}$

The second difference is that we update the BN parameters  $(\theta, \lambda)$  by their moving averages with respect to diminishing  $\alpha^{(m)}$ . The original BN algorithm can be recovered by setting  $\alpha^{(m)} = 1$  for every  $m$ . After introducing diminishing  $\alpha^{(m)}$ ,  $\lambda^{(m)}$  and hence the output of the BN layer is determined by the history of all past data records, instead of those solely in the last batch. Thus, the output of the BN layer becomes more general that better reflects the distribution of the entire dataset. We use two strategies to decide the values of  $\alpha^{(m)}$ . One is to use a constant smaller than 1 for all  $m$ , and the other one is to decay the  $\alpha^{(m)}$  gradually, such as  $\alpha^{(m)} = 1 / m$ .

In our numerical experiment, we show that Algorithm 1 outperforms the original BN algorithm, where both are based on SG and non-linear activation functions with many layers FNN and CNN models.

# 4 GENERAL CASE

The main purpose of our work is to show that Algorithm 1 converges. In the general case, we focus on the nonconvex objective function.

# 4.1 ASSUMPTIONS

Here are the assumptions we used for the convergence analysis.

Assumption 1 (Lipschitz continuity on  $\theta$  and  $\lambda$ ). For every  $i$  we have

$$
\left\| \nabla f _ {i} (X: \tilde {\theta}, \lambda) - \nabla f _ {i} (X: \hat {\theta}, \lambda) \right\| _ {2} \leq \bar {L} \| \tilde {\theta} - \hat {\theta} \| _ {2}, \forall \tilde {\theta}, \hat {\theta}, \lambda , X. \tag {7}
$$

$$
\left\| \nabla_ {W _ {1, j, \cdot}} f _ {i} (X: \tilde {\theta}, \lambda) - \nabla_ {W _ {1, j, \cdot}} f _ {i} (X: \hat {\theta}, \lambda) \right\| _ {2} \tag {8}
$$

$$
\leq \bar {L} \| \hat {W} _ {1, j, \cdot} - \hat {W} _ {1, j, \cdot} \| _ {2}, \forall \lambda , \tilde {\theta}, \hat {\theta}, X, j \in \{1, \dots , D _ {1} \}.
$$

$$
\left\| \nabla f _ {i} (X: \theta , \tilde {\lambda}) - \nabla f _ {i} (X: \theta , \hat {\lambda}) \right\| _ {2} \leq \bar {L} \| \tilde {\lambda} - \hat {\lambda} \| _ {2}, \tag {9}
$$

$$
\forall \theta , \tilde {\lambda}, \hat {\lambda}, X, j \in \{1, \dots , D _ {1} \}.
$$

Noted that the Lipschitz constants associated with each of the above inequalities are not necessarily the same. Here  $\bar{L}$  is an upper bound for these Lipschitz constants for simplicity.

Assumption 2 (bounded parameters). Sets  $P$  and  $Q$  are compact set, where  $\theta \in P$  and  $\lambda \in Q$ . Thus, there exists a constant  $M$  that weights  $W$  and parameters  $\lambda$  are bounded element-wise by this constant  $M$ .

$$
\left\| W _ {1} \right\| \preceq M a n d \left\| W _ {2} \right\| \preceq M a n d \left\| \lambda \right\| \preceq M.
$$

This also implies that the updated  $\theta, \lambda$  in Algorithm 1 remain in  $P$  and  $Q$ , respectively.

Assumption 3 (diminishing update on  $\theta$ ). The stepsizes of  $\theta$  update satisfy

$$
\sum_ {m = 1} ^ {\infty} \eta^ {(m)} = \infty \text {a n d} \sum_ {m = 1} ^ {\infty} \left(\eta^ {(m)}\right) ^ {2} <   \infty . \tag {10}
$$

This is a common assumption for diminishing step sizes in optimization problems.

Assumption 4 (Lipschitz continuity of  $l_{i}(\cdot)$ ). Assume the loss functions  $l_{i}(\cdot)$  for every  $i$  is continuously differentiable. It implies that there exists  $\hat{M}$  such that

$$
\left\| l _ {i} (x) - l _ {i} (y) \right\| \leq \hat {M} \| x - y \|, \forall x, y.
$$

Assumption 5 (existence of a stationary point). There exists a stationary point  $(\theta^{*},\lambda^{*})$  such that  $\| \nabla \bar{f} (\theta^{*},\lambda^{*})\| = 0$

We note that all these are standard assumptions in convergence proofs. We also stress that Assumption 4 does not directly imply 1. Since we assume that  $P$  and  $Q$  are compact, then Assumptions 1, 4 and 5 hold for many standard loss function such as softmax and MSE.

Assumption 6 (Lipschitz at activation function). The activation function  $a(\cdot)$  is Lipschitz with constant  $k$ :

$$
\left| a (x) \right| \leq k \| x \| \tag {11}
$$

Since for all activation function there is  $a(0) = 0$ , the condition is equivalent to  $|a(x) - a(0)| \leq k\| x - 0\|$ . We note that this assumption works for many popular choices of activation functions, such as ReLU and LeakyReLU.

# 4.2 CONVERGENCE ANALYSIS

We first have the following lemma specifying sufficient conditions for  $\lambda$  to converge. Proofs for main steps are given in the Appendix.

Theorem 7 Under Assumptions 1, 2, 3 and 6, if  $\{\alpha^{(m)}\}$  satisfies

$$
\sum_ {m = 1} ^ {\infty} \alpha^ {(m)} <   \infty a n d \sum_ {m = 1} ^ {\infty} \sum_ {n = 1} ^ {m} \alpha^ {(m)} \eta^ {(n)} <   \infty ,
$$

then sequence  $\{\lambda^{(m)}\}$  converges to  $\bar{\lambda}$ .

We give a discussion of the above conditions for  $\alpha^{(m)}$  and  $\eta^{(m)}$  at the end of this section. With the help of Theorem 7, we can show the following convergence result.

Lemma 8 Under Assumptions 4, 5 and the assumptions of Theorem 7, when

$$
\sum_ {m = 1} ^ {\infty} \sum_ {i = m} ^ {\infty} \sum_ {n = 1} ^ {i} \alpha^ {(i)} \eta^ {(n)} <   \infty \quad a n d \quad \sum_ {m = 1} ^ {\infty} \sum_ {n = m} ^ {\infty} \alpha^ {(n)} <   \infty , \tag {12}
$$

we have

$$
\lim  _ {M \rightarrow \infty} \sum_ {m = 1} ^ {M} \eta^ {(m)} \| \nabla \bar {f} \left(\theta^ {(m)}, \bar {\lambda}\right) \| _ {2} ^ {2} <   \infty . \tag {13}
$$

This result is similar to the classical convergence rate analysis for the non-convex objective function with diminishing step sizes, which can be found in Bottou et al. (2016).

Lemma 9 Under the assumptions of Lemma 8, we have

$$
\lim  _ {m \rightarrow \infty} \inf  _ \theta^ {(m)}, \bar {\lambda}) \| _ {2} ^ {2} = 0. \tag {14}
$$

This theorem states that for the full gradient method with diminishing stepsizes the gradient norms cannot stay bounded away from zero. The following result characterizes more precisely the convergence property of Algorithm 1.

Lemma 10 Under the assumptions stated in Lemma 8, we have

$$
\lim  _ {m \rightarrow \infty} \| \nabla \bar {f} \left(\theta^ {(m)}, \bar {\lambda}\right) \| _ {2} ^ {2} = 0. \tag {15}
$$

Our main result is listed next.

Theorem 11 Under the assumptions stated in Lemma 8, we have

$$
\lim  _ {m \rightarrow \infty} \| \nabla \bar {f} \left(\theta^ {(m)}, \lambda^ {(m)}\right) \| _ {2} ^ {2} = 0. \tag {16}
$$

We cannot show that  $\{\theta^{(m)}\}$ 's converges (standard convergence proofs are also unable to show such a stronger statement). For this reason, Theorem 11 does not immediately follow from Lemma 10 together with Theorem 7. The statement of Theorem 11 would easily follow from Lemma 10 if the convergence of  $\{\theta^{(m)}\}$  is established and the gradient being continuous.

Considering the cases  $\eta^{(m)} = O\left(\frac{1}{m^k}\right)$  and  $\alpha^{(m)} = O\left(\frac{1}{m^h}\right)$ . We show in the Appendix that the set of sufficient and necessary conditions to satisfy the assumptions of Theorem 7 are  $h > 1$  and  $k \geq 1$ . The set of sufficient and necessary conditions to satisfy the assumptions of Lemma 8 are  $h > 2$  and  $k \geq 1$ . For example, we can pick  $\eta^{(m)} = O\left(\frac{1}{m}\right)$  and  $\alpha^{(m)} = O\left(\frac{1}{m^{2.001}}\right)$  to achieve the above convergence result in Theorem 11.

# 5 COMPUTATIONAL EXPERIMENTS

We conduct the computational experiments with Theano and Lasagne on a Linux server with a Nvidia Titan-X GPU. We use MNIST LeCun et al. (1998), CIFAR-10 Krizhevsky & Hinton (2009) and Network Intrusion (NI) kdd (1999) datasets to compare the performance between DBN and the original BN algorithm. For the MNIST dataset, we use a four-layer fully connected FNN  $(784 \times 300 \times 300 \times 10)$  with the ReLU activation function and for the NI dataset, we use a four-layer fully connected FNN  $(784 \times 50 \times 50 \times 10)$  with the ReLU activation function. For the CIFAR-10 dataset, we use a reasonably complex CNN network that has a structure of (Conv-Conv-MaxPool-Dropout-Conv-Conv-MaxPool-Dropout-FC-Dropout-FC), where all four convolution layers and the first fully connected layers are batch normalized. We use the softmax loss function and  $l_{2}$  regularization with for all three models. All the trainable parameters are randomly initialized before training. For all 3 datasets, we use the standard epoch/minibatch setting with the minibatch size of 100, i.e., we do not compute the full gradient and the statistics are over the minibatch. We use AdaGrad Duchi, John and Hazan, Elad and Singer (2011) to update the learning rates  $\eta^{(m)}$  for trainable parameters, starting from 0.01.

We use two different strategies to decide the values of  $\alpha^{(m)}$  in DBN: constant values of  $\alpha^{(m)}$  and diminishing  $\alpha^{(m)}$  where  $\alpha^{(m)} = 1 / m$  and  $\alpha^{(m)} = 1 / m^2$ . We test the choices of constant  $\alpha^{(m)} \in \{1,0.75,0.5,0.25,0.1,0.01,0.001,0\}$ .

![](images/8e5050ed546549a8da80e4b20f86497c8c052bbba790715afe1ccbc7b71da572.jpg)  
(a)

![](images/c1b2ddb44489c5cfc57b26e780ab86da2b37d34177cb4e428dad7ad8a579ab17.jpg)  
(b)

![](images/f8ea3d01c0328428ff0f011d385a84997472c09d2914d131e7f882ff6280d789.jpg)  
(c)  
Figure 2: Comparison of predicted accuracy on test datasets for different choices of  $\alpha^{(m)}$ . From left to right are FNN on MNIST, FNN on NI and CNN on CIFAR-10.

![](images/8a1dbb8a8000bebd83c41a93121f9ebf3dc1e55ff987656e98862ccdb2940248.jpg)  
(a)

![](images/3571dd78adfacf6d6088ea1e00c6701f657dd2fc590c98c3df271a3e0d6151b2.jpg)  
(b)

![](images/2fb1de4d4225ef52fe5d5580275fbd7ed5cd16f128ec83779744e52ef4d863cd.jpg)  
(c)  
Figure 3: Comparison of predicted accuracy on test datasets for the most efficient choices of  $\alpha^{(m)}$ . From left to right are FNN on MNIST, FNN on MI and CNN on CIFAR-10.

![](images/12d25e41c2e45eac1f2edac475f6d0d4c8c3f6c66cb0513f17f80e8c3c0939d1.jpg)  
(a)

![](images/18602aeb3b2fd32962adefd299d4abfe6b6148705966e783ca891da357c5d99d.jpg)  
(b)  
Figure 4: Comparison of the convergence of the loss function value on the validation set for different choices of  $\alpha^{(m)}$ . From left to right are FNN on MNIST, FNN on NI and CNN on CIFAR-10.

![](images/272596c0fe4bd04be8ec582f403698e16c74906d30de9d010c5aa0e823d5eafc.jpg)  
(c)

We test all the choices of  $\alpha^{(m)}$  with the performances presented in Figure 2. Figure 2 shows that all the non-zero choices of  $\alpha^{(m)}$  converge properly. The algorithms converge without much difference even when  $\alpha^{(m)}$  in DBN is very small, e.g.,  $1 / m^2$ . However, if we select  $\alpha^{(m)} = 0$ , the algorithm is erratic. Besides, we observe that all the non-zero choices of  $\alpha^{(m)}$  converge at a similar rate. The fact that DBN keeps the batch normalization layer stable with a very small  $\alpha^{(m)}$  suggests that the BN parameters do not have to be depended on the latest minibatch, i.e., the original BN.

We compare a selected set of the most efficient choices of  $\alpha^{(m)}$  in Figures 3 and 4. They show that DBN with  $\alpha^{(m)} < 1$  is more stable than the original BN algorithm. The variances with respect to epochs of the DBN algorithm are smaller than those of the original BN algorithms in each figure.

Table 1: Best results for different choices of  ${\alpha }^{\left( m\right) }$  on each dataset,showing the top three with a heat map.  

<table><tr><td rowspan="2">Model</td><td colspan="3">Test Error</td></tr><tr><td>MNIST</td><td>NI</td><td>CIFAR-10</td></tr><tr><td>α(m) = 1</td><td>2.70%</td><td>7.69%</td><td>17.31%</td></tr><tr><td>α(m) = 0.75</td><td>1.91%</td><td>7.37%</td><td>17.03%</td></tr><tr><td>α(m) = 0.5</td><td>1.84%</td><td>7.46%</td><td>17.11%</td></tr><tr><td>α(m) = 0.25</td><td>1.91%</td><td>7.24%</td><td>17.00%</td></tr><tr><td>α(m) = 0.1</td><td>1.90%</td><td>7.36%</td><td>17.10%</td></tr><tr><td>α(m) = 0.01</td><td>1.94%</td><td>7.47%</td><td>16.82%</td></tr><tr><td>α(m) = 0.001</td><td>1.95%</td><td>7.43%</td><td>16.28%</td></tr><tr><td>α(m) = 1/m</td><td>2.10%</td><td>7.45%</td><td>17.26%</td></tr><tr><td>α(m) = 1/m2</td><td>2.00%</td><td>7.59%</td><td>17.23%</td></tr><tr><td>α(m) = 0</td><td>24.27%</td><td>26.09%</td><td>79.34%</td></tr></table>

Table 1 shows the best result obtained from each choice of  $\alpha^{(m)}$ . Most importantly, it suggests that the choices of  $\alpha^{(m)} = 1/m$  and  $1/m^2$  perform better than the original BN algorithm. Besides, all the constant less-than-one choices of  $\alpha^{(m)}$  perform better than the original BN, showing the importance of considering the mini-batch history for the update of the BN parameters. The BN algorithm in each figure converges to similar error rates on test datasets with different choices of  $\alpha^{(m)}$  except for the  $\alpha^{(m)} = 0$  case. Among all the models we tested,  $\alpha^{(m)} = 0.25$  is the only one that performs top 3 for all three datasets, thus the most robust choice.

To summarize, our numerical experiments show that the DBN algorithm outperforms the original BN algorithm on the MNIST, NI and CIFAT-10 datasets with typical deep FNN and CNN models.

Future Directions. On the analytical side, we believe an extension to more than 2 layers is doable with significant augmentations of the notation. A stochastic gradient version is likely to be much more challenging to analyze. A second open question concerns more general activation functions. It would be interesting to analyze other activation functions, such as Sigmoid, that do not apply to our current assumptions.

# REFERENCES

KDD Cup 1999 Data, 1999. URL http://www.kdd.org/kdd-cup/view/ kdd-cup-1999/Data.  
Devansh Arpit, Yingbo Zhou, Bhargava U. Kota, and Venu Govindaraju. Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks. In International Conference on Machine Learning, volume 48, pp. 11, 2016.  
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer Normalization. arXiv preprint arXiv:1607.06450, 2016.  
Dimitri P. Bertsekas. Incremental gradient, subgradient, and proximal methods for convex optimization: A Survey. Optimization for Machine Learning, 2010(3):1-38, 2011.  
Dimitri P. Bertsekas and John N. Tsitsiklis. Gradient Convergence in Gradient Methods with Errors. SIAM Journal on Optimization, 10:627-642, 2000.  
Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization Methods for Large-Scale Machine Learning. arXiv preprint arXiv:1606.04838, 2016.  
Tim Cooijmans, Nicolas Ballas, César Laurent, and Aaron Courville. Recurrent Batch Normalization. arXiv preprint arXiv:1603.09025, 2016.  
Yoram Duchi, John and Hazan, Elad and Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
L. Grippo. A Class of Unconstrained Minimization Methods for Neural Network Training. Optimization Methods and Software, 4(2):135-150, 1994.  
Caglar Güçehre and Yoshua Bengio. Knowledge Matters: Importance of Prior Information for Optimization. Journal of Machine Learning Research, 17(8):1-32, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In Computer Vision and Pattern Recognition, pp. 770-778, dec 2016.  
Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In International Conference on Machine Learning, pp. 448-456, 2015.  
Alex Krizhevsky and Geoffrey E. Hinton. Learning Multiple Layers of Features from Tiny Images. PhD thesis, 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet Classification with Deep Convolutional Neural Networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Yann LeCun, Léon Bottou, and Yoshua Bengio. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Genevieve B. Orr and Klaus-Robert Müller. Neural Networks: Tricks of the Trade. Springer, New York, 2003.  
B. T. Polyak. Introduction to optimization. Translations series in mathematics and engineering. Optimization Software, 1987.  
B. T. Polyak and Y. Z. Tsypkin. Pseudogradient Adaption and Training Algorithms. Automation and Remote Control, 34:45-67, 1973.  
Tim Salimans and Diederik P. Kingma. Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks. In Advances in Neural Information Processing Systems, pp. 901-901, 2016.
