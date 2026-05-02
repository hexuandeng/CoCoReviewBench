# BLOCK-NORMALIZED GRADIENT METHOD: AN EMPIRICAL STUDY FOR TRAINING DEEP NEURAL NETWORK

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this paper, we propose a generic and simple strategy for utilizing stochastic gradient information in optimization. The technique essentially contains two consecutive steps in each iteration: 1) computing and normalizing each block (layer) of the mini-batch stochastic gradient; 2) selecting appropriate step size to update the decision variable (parameter) towards the negative of the block-normalized gradient. We conduct extensive empirical studies on various non-convex neural network optimization problems, including multi layer perceptron, convolution neural networks and recurrent neural networks. The results indicate the block-normalized gradient can help accelerate the training of neural networks. In particular, we observe that the normalized gradient methods having constant step size with occasionally decay, such as SGD with momentum, have better performance in the deep convolution neural networks, while those with adaptive step sizes, such as Adam, perform better in recurrent neural networks. Besides, we also observe this line of methods can lead to solutions with better generalization properties, which is confirmed by the performance improvement over strong baselines.

# 1 INTRODUCTION

Continuous optimization is a core technique for training non-convex sophisticated machine learning models such as deep neural networks (Bengio, 2009). Compared to convex optimization where a global optimal solution is expected, non-convex optimization usually aims to find a stationary point or a local optimal solution of an objective function by iterative algorithms. Among a large volume of optimization algorithms, first-order methods, which only use (stochastic) gradient information of objective functions to update solution, are most widely used due to its relatively low requirement on memory space and computation time.

In this paper, we are particularly interested in solving the deep neural network training problems with first order method. However, compared to other non-convex problems, deep neural network training has the following challenges:

1. Gradient computation is expensive. A full gradient is usually difficult to compute, especially when applied to machine learning problems over big data, because its calculation requires going through the whole dataset.  
2. Gradient may be vanishing and/or exploding. The gradient may become very small over some plateaus regions while sharply becoming large on the cliffs of objective function. As the number of layers in the neural network increases, this phenomenon of vanishing or exploding gradients becomes more severe so that the iterative solution will converge very slowly or diverge quickly.

These two challenges indicate that for each iteration, stochastic gradient might be the best practical first order information we can get. Once we obtain this information, we should develop certain mechanism to avoid its exploding or vanishing phenomenon and eventually ensure faster convergence.

In this paper we aim to solve those challenges from the two key ingredients of first-order optimization algorithm per iteration, i.e., an updating direction and a step size (a.k.a. learning rate). The

updating direction indicates where the next iterated point should move to and the step size determines how far the move should end up with.

We propose to use stochastic normalized gradient which is constructed by dividing the (stochastic) gradient by its norm. Compared to the regular gradient, normalized gradient only provides an updating direction but does not incorporate the local steepness of the objective through its magnitude, which help to control the change of the solution through a well-designed step length. Intuitively, as it constrains the magnitude of the gradient to be 1, it should to some extent prevent the gradient vanishing or exploding phenomenon. In fact, as showed in (Hazan et al., 2015; Levy, 2016), normalized gradient descent (NGD) methods are more numerically stable and have better theoretical convergence properties than the regular gradient descent method in non-convex optimization.

Once the updating direction is determined, step size is the next important component in the design of first-order methods. While for convex problems there are some well studied strategies to find a stepsize ensuring convergence, for non-convex optimization, the choice of step size is more difficult and critical as it may either enlarge or reduce the impact of the aforementioned vanishing or exploding gradients. When a normalized gradient is used, the step size fully controls the progress of the solution so that the convergence is possible only when the step size is carefully designed.

Among different choices of step sizes, the adaptive feature-dependent step size has attracted lots of attention due to its great performance when applied to training deep neural networks (Duchi et al., 2011; Kingma & Ba, 2014). Different from the standard step-size rule which multiplies a same number to each coordinate of gradient, the adaptive feature-dependent step-size rule multiplies different numbers to coordinates of gradient so that different parameters in the learning model can be updated in different paces. For example, the adaptive step size invented by (Duchi et al., 2011) is constructed by aggregating each coordinates of historical gradients. As discussed by (Duchi et al., 2011), this method can dynamically incorporate the frequency of features in the step size so that frequently occurring features will have a small step size while infrequent features have a long step size. The similar adaptive step size is proposed in (Kingma & Ba, 2014) but the historical gradients are integrated into feature-dependent step size by a different weighting scheme. While the adaptive stepsize is useful in the recurrent neural network, as we observe in the experiments, we also find that methods having constant stepsize works better in the deep convolution neural networks.

In this paper, we propose a generic framework using the mini-batch stochastic normalized gradient as the updating direction (like (Hazan et al., 2015; Levy, 2016)) and the step size is either constant or adaptive to each coordinate as in (Duchi et al., 2011; Kingma & Ba, 2014). In this way, we hope the resulting methods can benefit from both techniques and generate a even better performance than the existing methods. Our framework starts with computing regular mini-batch stochastic gradient, which is immediately normalized. The normalized version is then plugged in the current popular adaptive step size methods, such as Adam (Kingma & Ba, 2014) and AdaGrad (Duchi et al., 2011), or the constant stepsize with occasional decay, such as  $\mathrm{SGD + }$  momentum. The numerical results show that normalized gradient always helps to improve the performance of the original methods especially when the network structure is deep. It seems to be the first thorough empirical study on various types of neural networks with this normalized gradient idea. Besides, although we focus our empirical studies on deep learning where the objective is highly non-convex, we also provide a convergence proof under this framework when the problem is convex and the stepsize is adaptive in the appendix.

The rest of the paper is organized as follows. In Section 2, we briefly go through the previous work that are related to ours. In Section 3, we formalize the problem to solve and propose the generic algorithm framework. In Section 4, we conduct comprehensive experimental studies to compare the performance of different algorithms on various neural network structures. We conclude the paper in Section 5. Finally, in the appendix, we provide a concrete example of this type of algorithm and show its convergence property under the convex setting.

# 2 RELATED WORK

A pioneering work on normalized gradient descent (NGD) method was by Nesterov (Nesterov, 1984) where it was shown that NGD can find a  $\epsilon$ -optimal solution within  $O\left(\frac{1}{\epsilon^2}\right)$  iterations when the objective function is quasi-convex. Kiwiel (Kiwiel, 2001) and Hazan et al (Hazan et al., 2015)

extended NGD for upper semi-continuous quasi-convex objective functions and local-quasi-convex objective functions, respectively, and achieved the same iteration complexity. Moreover, Hazan et al (Hazan et al., 2015) showed that NGD's iteration complexity can be reduced to  $O\left(\frac{1}{\epsilon}\right)$  if the objective function is local-quasi-convex and locally-smooth. A stochastic NGD algorithm is also proposed by Hazan et al (Hazan et al., 2015) which, if a mini-batch is used to construct the stochastic normalized gradient in each iteration, finds  $\epsilon$ -optimal solution with a high probability for locally-quasi-convex functions within  $O\left(\frac{1}{\epsilon^2}\right)$  iterations. Levy (Levy, 2016) proposed a Saddle-Normalized Gradient Descent (Saddle-NGD) method, which adds a zero-mean Gaussian random noise to the stochastic normalized gradient periodically. When applied to strict-saddle functions with some additional assumption, it is shown (Levy, 2016) that Saddle-NGD can evade the saddle points and find a local minimum point approximately with a high probability.

Analogous yet orthogonal to the gradient normalization ideas have been proposed for the deep neural network training. For example, batch normalization (Ioffe & Szegedy, 2015) is used to address the internal covariate shift phenomenon in the during deep learning training. It benefits from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Weight normalization (Salimans & Kingma, 2016), on the other hand, aims at a reparameterization of the weight vectors that decouples the length of those weight vectors from their direction. Recently (Neyshabur et al., 2015) proposes to use path normalization, an approximate path-regularized steepest descent with respect to a path-wise regularizer related to max-norm regularization to achieve better convergence than vanilla SGD and AdaGrad. Perhaps the most related idea to ours is Gradient clipping. It is proposed in (Pascanu et al., 2013) to avoid the gradient explosion, by pulling the magnitude of a large gradient to a certain level. However, this method does not do anything when the magnitude of the gradient is small.

Adaptive step size has been studied for years in the optimization community. The most celebrated method is the line search scheme. However, while the exact line search is usually computational infeasible, the inexact line search also involves a lot of full gradient evaluation. Hence, they are not suitable for the deep learning setting. Recently, provably correct algorithms with adaptive step sizes in the convex scenario start to be applied to the non-convex neural network training, such as AdaGrad (Duchi et al., 2012), Adam (Kingma & Ba, 2014) and RMSProp. However they directly use the unnormalized gradient, which is different from our framework. To the best of our knowledge, our work seems to be the first study on combining normalized gradient with adaptive stepsize for deep neural network training.

# 3 ALGORITHM FRAMEWORK

In this section, we present our algorithm framework to solve the following general problem:

$$
\min  _ {x \in \mathbb {R} ^ {d}} f (x) = \mathbb {E} (F (x, \xi)), \tag {1}
$$

where  $x = (x^{1}, x^{2}, \ldots, x^{B}) \in \mathbb{R}^{d}$  with  $x^{i} \in \mathbb{R}^{d_{i}}$  and  $\sum_{i=1}^{B} d_{i} = d$ ,  $\xi$  is a random variable following some distribution  $\mathbb{P}$ ,  $F(\cdot, \xi)$  is a loss function for each  $\xi$  and the expectation  $\mathbb{E}$  is taken over  $\xi$ . Now our goal is to minimize the objective function  $f$  over  $x$ , where  $x$  can be the parameters of a machine learning model when (1) corresponds to a training problem. Here, the parameters are partitioned into  $B$  blocks. The problem of training a neural network is an important instance of (1), where each block of parameters  $x^{i}$  can be viewed as the parameters associated to the  $i$ th layer in the network.

We propose the generic optimization framework in Algorithm 1. In iteration  $t$ , it firstly computes the partial (sub)gradient  $F_{i}^{\prime}(x_{t},\xi_{t})$  of  $F$  with respect to  $x^{i}$  for  $i = 1,2,\ldots$ , at  $x = x_{t}$  with a mini-batch data  $\xi_{t}$ , and then normalizes it to get a partial direction  $g_{t}^{i} = \frac{F_{i}^{\prime}(x_{t},\xi_{t})}{\|F_{i}^{\prime}(x_{t},\xi_{t})\|_{2}}$ . We define  $g_{t} = (g_{t}^{1},g_{t}^{2},\dots ,g_{t}^{B})$ . The next is to find  $d$  adaptive step sizes  $\tau_t\in \mathbb{R}^d$  with each coordinate of  $\tau_{t}$  corresponding to a coordinate of  $x$ . We also partition  $\tau_{t}$  in the same way as  $x$  so that  $\tau_{t} = (\tau_{t}^{1},\tau_{t}^{2},\dots ,\tau_{t}^{B})\in \mathbb{R}^{B}$  with  $\tau_t^i\in \mathbb{R}^{d_i}$ . We use  $\tau_{t}$  as step sizes to update  $x_{t}$  to  $x_{t + 1}$  as  $x_{t + 1} = x_{t} - \tau_{t}\circ g_{t}$ , where  $\circ$  represents coordinate-wise (Hadamard) product. In fact, our framework can be customized to most of existing first order methods with fixed or adaptive step sizes, such as SGD, AdaGrad(Duchi et al., 2011), RMSProp and Adam(Kingma & Ba, 2014), by adopting their step size rules respectively.

Algorithm 1 Generic Block-Normalized Gradient (BNG) Descent  
1: Choose  $x_{1} \in \mathbb{R}^{d}$ .  
2: for  $t = 1, 2, \ldots$ , do  
3: Sample a mini-batch of data  $\xi_{t}$  and compute the partial stochastic gradient  $g_{t}^{i} = \frac{F_{i}^{\prime}(x_{t}, \xi_{t})}{\|F_{i}^{\prime}(x_{t}, \xi_{t})\|_{2}}$   
4: Let  $g_{t} = (g_{t}^{1}, g_{t}^{2}, \ldots, g_{t}^{B})$  and choose step sizes  $\tau_{t} \in \mathbb{R}^{d}$ .  
5:  $x_{t+1} = x_{t} - \tau_{t} \circ g_{t}$   
6: end for

# 4 NUMERICAL EXPERIMENTS

Basic Experiment Setup In this section, we conduct comprehensive numerical experiments on different types of neural networks. The algorithms we are testing are SGD with Momentum (SGDM), AdaGrad (Duchi et al., 2013), Adam (Kingma & Ba, 2014) and their block-normalized gradient counterparts, which are denoted with suffix "NG". Specifically, we partition the parameters into block as  $x = (x_{1}, x_{2}, \ldots, x_{B})$  such that  $x_{i}$  corresponds to the vector of parameters (including the weight matrix and the bias/intercept coefficients) used in the  $i$ th layer in the network.

Our experiments are on four diverse tasks, ranging from image classification to natural language processing. The neural network structures under investigation include multi layer perceptron, long-short term memory and convolution neural networks.

To exclude the potential effect that might be introduced by advanced techniques, in all the experiments, we only adopt the basic version of the neural networks, unless otherwise stated. The loss functions for classifications are cross entropy, while the one for language modeling is log perplexity. Since the computational time is proportional to the epochs, we only show the performance versus epochs. Those with running time are similar so we omit them for brevity. For all the algorithms, we use their default settings. More specifically, for Adam/AdamNG, the initial step size scale  $\alpha = 0.001$ , first order momentum  $\beta_{1} = 0.9$ , second order momentum  $\beta_{2} = 0.999$ , the parameter to avoid division of zero  $\epsilon = 1e^{-8}$ ; for AdaGrad/AdaGradNG, the initial step size scale is 0.01.

# 4.1 MNIST IMAGE CLASSIFICATION WITH FEEDFORWARD NEURAL NETWORK

The first network structure we are going to test upon is the Multi Layer Perceptron (MLP). We will adopt the handwritten digit recognition data set MNIST $^1$ Lecun et al. (1998), in which, each data is an image of hand written digits from  $\{1,2,3,4,5,6,7,8,9,0\}$ . There are 60k training and 10k testing examples and the task is to tell the right number contained in the test image. Our approach is applying MLP to learn an end-to-end classifier, where the input is the raw  $28\times 28$  images and the output is the label probability. The predicted label is the one with the largest probability. In each middle layer of the MLP, the hidden unit number are 100, and the first and last layer respectively contain 784 and 10 units. The activation functions between layers are all sigmoid and the batch size is 100 for all the algorithms.

We choose different numbers of layer from  $\{6,12,18\}$ . The results are shown in Figure 1. Each column of the figures corresponds to the training and testing objective curves of the MLP with a given layer number. From left to right, the layer numbers are respectively 6, 12 and 18. We can see that, when the network is as shallow as containing 6 layers, the normalized stochastic gradient descent can outperform its unnormalized counterpart, while the Adam and AdaGrad are on par with or even slightly better than their unnormalized versions. As the networks become deeper, the acceleration brought by the gradient normalization turns more significant. For example, starting from the second column, AdamNG outperforms Adam in terms of both training and testing convergence. In fact, when the network depth is 18, the AdamNG can still converge to a small objective value while Adam gets stuck from the very beginning. We can observe the similar trend in the comparison between AdaGrad (resp. SGDM) and AdaGradNG (resp. SGDMNG). On the other hand, the algorithms with adaptive step sizes can usually generate a stable learning curve. For example, we can see from the last two columns that SGDNG causes significant fluctuation in both training and testing

curves. Finally, under any setting, AdamNG is always the best algorithm in terms of convergence performance.

![](images/d8cde47f26ec4ed6bf40afead568cc658324fdc58d2fb8aca7c8450ded8ce9a9.jpg)

![](images/eee076189ab7f310f9720160a387e7818563677a17eec33964dbbc46658e6258.jpg)

![](images/f8c39a306a1c15d3db9e41e374863f44c0ec77e7484935774799d7c2645ced5c.jpg)

![](images/62ba64cc7bebbd5fd2b1489053901a6cdc0c26963e96019cfe120dbcd44e0656.jpg)  
Figure 1: The training and testing objective curves on MNIST dataset with multi layer perceptron. From left to right, the layer numbers are 6, 12 and 18 respectively. The first row is the training curve and the second is testing.

![](images/f2865704583a681b1c784c21c4b2a1966d2f1c91f8d85502e9d316993b2ed3b4.jpg)

![](images/4994b181346dfe69603d97deed25ccc35e63f4df4ef278a4c9fa6373205efacf.jpg)

# 4.2 RESIDUAL NETWORK ON CIFAR 10 AND CIFAR100

In the following section, we compare the methods on CIFAR (both CIFAR-10 and CIFAR-100) datasets over the residual network He et al. (2016). CIFAR-10 consists of 50,000 training images and 10,000 test images from 10 classes, while CIFAR-100 from 100 classes. Each input image consists of  $32 \times 32$  pixels. The dataset was preprocessed as described in He et al. (2016) by subtracting the means and dividing the variance for each channel. We follow the simple data augmentation that 4 pixels are padded on each side, and a  $32 \times 32$  crop is randomly sampled from the padded image or its horizontal flip as described in He et al. (2016).

We compare two major methods, namely SGD and ADAM, who respectively represents the constant step size with occasional decay and adaptive step size. All the algorithms are combined with batch normalization technique. We compare different gradient processing schemes under these two methods, where "BN" is the plain batch normalization without any processing, "Grad clipping" is the standard gradient clipping method, "NG<sub>UNIT</sub>" is the default normalized gradient method, while "NG" (we abuse the notation a bit) is a variant of the NG<sub>UNIT</sub> method, i.e. NG = NG<sub>UNIT</sub> × Norm of variable × ratio. We empirically observe that NG has better performance than NG<sub>UNIT</sub>.

We follow the exactly same experimental protocol as described in He et al. (2016) and adopt the publicly available Torch implementation<sup>2</sup> for residual network. That is (1) For training, SGD is used with momentum of 0.9, weight decay of 0.0001 and mini-batch size of 128. The initial learning rate is set as 0.1 and dropped by a factor of 0.1 at 80, 120 with a total training of 160 epochs; (2) The weight initialization method is used as He et al. (2015).

We use the residual network architecture with varied depths  $L = \{20, 32, 44, 56, 110\}$  and over CIFAR-10 and CIFAR100 dataset. For the extra hyper-parameters: thresh of Clipping, and ratio of NG, we chose the best heyper-paerameter from the 56 layer residual network. Particularly, for the thresh of Clipping, the search values are 0.05, 0.1, 0.5, 1, 5, and the best value is 0.1. For the ratio of NG, the search values are 0.01, 0.02, 0.05, and the best value is 0.1

The results are shown in Figure 2 and Table 1 and 2. We can see that the methods with normalized gradient converges much faster than those without, no matter the step size is constant or adaptive. We also observe that the normalized gradient method, especially the version "NG" has a better generalization properties.

![](images/a8a67eecb99a34a08dd50f79c598e6ab13693aa62e62f25c32d23e0709a0d210.jpg)  
Figure 2: The training and testing objective curves on CIFAR10 and CIFAR100 datasets with Res-110. Left: CIFAR10; Right: CIFAR100. The thick curves are the training while the thin are testing.

![](images/79318a0d94c111d07d1742f59162794aadbb3ca2b710477d255e9dc843f3674a.jpg)

<table><tr><td>Algorithm</td><td>Res-20</td><td>Res-32</td><td>Res-44</td><td>Res-56</td><td>Res-110</td></tr><tr><td colspan="6">SGD</td></tr><tr><td>BN</td><td>7.93</td><td>7.15</td><td>7.09</td><td>7.34</td><td>7.07</td></tr><tr><td>Grad Clipping</td><td>9.03</td><td>8.44</td><td>8.55</td><td>8.30</td><td>8.35</td></tr><tr><td>NGUNIT</td><td>7.82</td><td>7.09</td><td>6.60</td><td>6.59</td><td>6.28</td></tr><tr><td>NG</td><td>7.71</td><td>6.90</td><td>6.43</td><td>6.19</td><td>5.87</td></tr><tr><td colspan="6">ADAM</td></tr><tr><td>BN</td><td>9.14</td><td>8.33</td><td>7.794</td><td>7.33</td><td>6.75</td></tr><tr><td>Grad Clipping</td><td>10.18</td><td>9.18</td><td>8.89</td><td>9.24</td><td>9.96</td></tr><tr><td>NGUNIT</td><td>9.42</td><td>8.50</td><td>8.06</td><td>7.69</td><td>7.29</td></tr><tr><td>NG</td><td>8.52</td><td>7.62</td><td>7.28</td><td>7.04</td><td>6.71</td></tr></table>

Table 1: Error rates of ResNets with different depths on CIFAR 10.  

<table><tr><td>Algorithm</td><td>Res-20</td><td>Res-32</td><td>Res-44</td><td>Res-56</td><td>Res-110</td></tr><tr><td>SGD</td><td>32.28</td><td>30.62</td><td>29.96</td><td>29.07</td><td>28.79</td></tr><tr><td>Grad Clipping</td><td>35.06</td><td>34.49</td><td>33.36</td><td>34.00</td><td>33.38</td></tr><tr><td>NGUNIT</td><td>32.46</td><td>31.16</td><td>30.05</td><td>29.42</td><td>27.49</td></tr><tr><td>NG</td><td>31.43</td><td>29.56</td><td>28.92</td><td>28.48</td><td>26.72</td></tr></table>

Table 2: Error rates of ResNets with different depths on CIFAR 100.

# 4.3 34 LAYER PRE-RESIDUAL NETWORK FOR IMAGENET CLASSIFICATION

In this section, we further test our methods on ImageNet 2012 classification challenges, which consists of more than 1.2M images from 1,000 classes. We use the given 1.28M labeled images for training and the validation set with 50k images for testing. We employ the validation set as the test set, and evaluate the classification performance based on top-1 and top-5 error. The pre-activation version of ResNet is adopted in our experiments to perform the classification task.

We run our experiments on one GPU and use single scale and single crop test for simplifying discussion. We keep all the experiments settings the same as the publicly available Torch implementation  $^{3}$ , (that is, we apply stochastic gradient descent with momentum of 0.9, weight decay of 0.0001, and

set the initial learning rate to 0.1.). The exception is that we use mini-batch size of 64 and 50 training epochs considering the GPU memory limitations and training time costs. Regarding learning rate annealing, we use exponential decay to 0.001.

Due to the time-consuming nature of training the networks (which usually takes one week) in this experiment, we only compare SGD with our default  $\mathrm{NG}_{\mathrm{UNIT}}$  method on the testing error of the classification. From Table 3, we can see  $\mathrm{NG}_{\mathrm{UNIT}}$  has a non-trivial improvement on the testing error over SGD.

<table><tr><td>method</td><td>Top-1</td><td>Top-5</td></tr><tr><td>SGD</td><td>29.05</td><td>9.95</td></tr><tr><td>NGUNIT</td><td>28.43</td><td>9.57</td></tr></table>

Table 3: Top-1 and Top 5 error rates of ResNets on ImageNet classification.

# 4.4 LANGUAGE MODELING WITH RECURRENT NEURAL NETWORK

Now we move on to test the algorithms on Recurrent Neural Networks (RNN). In this section, we test the performance of the proposed algorithm on the word-level language modeling task with a popular type of RNN, i.e. single directional Long-Short Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997). The data set under use is Penn Tree Bank (PTB) (Marcus et al., 1993) data, which, after preprocessed, contains 929k training words, 73k validation and 82k test words. The vocabulary size is about 10k. The LSTM has 2 layers, each containing 200 hidden units. The word embedding has 200 dimensions which is trained from scratch. The batch size is 100. We vary the length of the backprop through time (BPTT) within the range  $\{40, 400, 1000\}$ . To prevent overfitting, we add a dropout regularization with rate 0.5 under all the settings.

The results are shown in Figure 3. The conclusions drawn from those figures are again similar to those in the last two experiments. However, the slightly different yet cheering observations is that the AdamNG is uniformly better than all the other competitors with any training sequence length. The superiority in terms of convergence speedup exists in both training and testing.

![](images/53382b51f58cc2f1e44becc78933c754e502189f1f9c8b98e03cf6cb4e5a67a3.jpg)

![](images/03ac73e0bb5a37c84529241d7c8a91182639a7abb7b2405377e14865b7bef22a.jpg)

![](images/019abb8acfb98c3cfb130636ee62d25658772196b61bc1377f80b8dc98a6029c.jpg)

![](images/b71b04f9a7c9ae64a01df4b7cbe2ef058f774031cf9bf0030ebb5011d76ee3ef.jpg)  
Figure 3: The training and testing objective curves on Penn Tree Bank dataset with LSTM recurrent neural networks. The first row is the training objective while the second is the testing. From left to right, the training sequence (BPTT) length are respectively 40, 400 and 1000. Dropout with 0.5 is imposed.

![](images/9d89a8092056c37db578d0cdd07dd4a155d4cd901feecb5ad56a0e6bf912a2dd.jpg)

![](images/32ec71d8c9c312ab55aef49d7915142414e3b1af5d5a1473be4719c30c17719b.jpg)

# 4.5 SENTIMENT ANALYSIS WITH CONVOLUTION NEURAL NETWORK

The task in this section is the sentiment analysis with convolution neural network. The dataset under use is Rotten Tomatoes<sup>4</sup> Pang & Lee (2005), a movie review dataset containing 10,662 documents, with half positive and half negative. We randomly select around  $90\%$  for training and  $10\%$  for validation. The model is a single layer convolution neural network that follows the setup of (Kim, 2014). The word embedding under use is randomly initialized and of 128-dimension.

For each algorithm, we run 150 epochs on the training data, and report the best validation accuracy in Table 4. The messages conveyed by the table is three-fold. Firstly, the algorithms using normalized gradient achieve much better validation accuracy than their unnormalized versions. Secondly, those with adaptive stepsize always obtain better accuracy than those without. This is easily seen by the comparison between Adam and SGDM. The last point is the direct conclusion from the previous two that the algorithm using normalized gradient with adaptive step sizes, namely AdamNG, outperforms all the remaining competitors.

<table><tr><td>Algorithm</td><td>AdamNG</td><td>Adam</td><td>AdaGradNG</td><td>AdaGrad</td><td>SGDMNG</td><td>SGDM</td></tr><tr><td>Validation Accuracy</td><td>77.11%</td><td>74.02%</td><td>71.95%</td><td>69.89%</td><td>71.95%</td><td>64.35%</td></tr></table>

Table 4: The Best validation accuracy achieved by the different algorithms.

# 5 CONCLUSION

In this paper, we propose a generic algorithm framework for first order optimization. It is particularly effective for addressing the vanishing and exploding gradient challenge in training with non-convex loss functions, such as in the context of convolutional and recurrent neural networks. Our method is based on normalizing the gradient to establish the descending direction regardless of its magnitude, and then separately estimating the ideal step size adaptively or constantly. This method is quite general and may be applied to different types of networks and various architectures. Although the primary application of the algorithm is deep neural network training, we provide a convergence for the new method under the convex setting in the appendix.

Empirically, the proposed method exhibits very promising performance in training different types of networks (convolutional, recurrent) across multiple well-known data sets (image classification, natural language processing, sentiment analysis, etc.). In general, the positive performance differential compared to the state of the art is most striking for very deep networks, as shown in our comprehensive experimental study.

# REFERENCES

Yoshua Bengio. Learning deep architectures for ai. Foundations and trends® in Machine Learning, 2(1):1-127, 2009.  
John Duchi, Michael I Jordan, and Brendan McMahan. Estimation, optimization, and parallelism when data is sparse. In NIPS, pp. 2832-2840, 2013.  
John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121-2159, 2011.  
John C Duchi, Alekh Agarwal, and Martin J Wainwright. Dual averaging for distributed optimization: convergence analysis and network scaling. Automatic Control, IEEE Transactions on, 57 (3):592-606, 2012.  
Elad Hazan, Kfir Y. Levy, and Shai Shalev-Shwartz. Beyond convexity: Stochastic quasi-convex optimization. In NIPS, pp. 1594–1602, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, pp. 448-456, 2015.  
Yoon Kim. Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882, 2014.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.  
Krzysztof C Kiwiel. Convergence and efficiency of subgradient methods for quasiconvex minimization. Mathematical programming, 90(1):1-25, 2001.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.  
Yann Lecun, Leon Bottou, Yoshua Bengio, and Patrick Haffner? Gradient-based learning applied to document recognition. In Proceedings of the IEEE, pp. 2278-2324, 1998.  
Kfir Y. Levy. The power of normalization: Faster evasion of saddle points. CoRR, abs/1611.04831, 2016. URL http://arxiv.org/abs/1611.04831.  
Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. Computational Linguistics, 19(2):313-330, 1993.  
Nesterov. Minimization methods for nonsmooth convex and quasiconvex functions. Matekon, 29: 519-531, 1984.  
Behnam Neyshabur, Ruslan Salakhutdinov, and Nathan Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In NIPS, pp. 2422-2430, 2015.  
Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the 43rd annual meeting on association for computational linguistics, pp. 115-124. Association for Computational Linguistics, 2005.  
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In ICML, pp. 1310-1318, 2013.  
Tim Salimans and Diederik P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In NIPS, 2016.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. URL http://arxiv.org/abs/1409.1556.
