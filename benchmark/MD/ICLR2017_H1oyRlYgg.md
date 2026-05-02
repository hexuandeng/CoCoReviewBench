# ON LARGE-BATCH TRAINING FOR DEEP LEARNING: GENERALIZATION GAP AND SHARP MINIMA

Nitish Shirish Keskar*

Northwestern University

Evanston, IL 60208

keskar.nitish@u.northwestern.edu

Dheevatsa Mudigere

Intel Corporation

Bangalore, India

dheevatsa.mudigere@intel.com

Jorge Nocedal

Northwestern University

Evanston, IL 60208

j-nocedal@northwestern.edu

Mikhail Smelyanskiy & Ping Tak Peter Tang

Intel Corporation

Santa Clara, CA 95054

{mikhail.smelyanskiy,peter.tang}@intel.com

# ABSTRACT

The stochastic gradient descent (SGD) method and its variants are algorithms of choice for many Deep Learning tasks. These methods operate in a small-batch regime wherein a fraction of the training data, say 32-512 data points, is sampled to compute an approximation to the gradient. It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, as measured by its ability to generalize. We investigate the cause for this generalization drop in the large-batch regime and present numerical evidence that supports the view that large-batch methods tend to converge to sharp minimizers of the training and testing functions—and as is well known, sharp minima lead to poorer generalization. In contrast, small-batch methods consistently converge to flat minimizers, and our experiments support a commonly held view that this is due to the inherent noise in the gradient estimation. We also discuss several empirical strategies that might help large-batch methods eliminate this generalization gap, and conclude with a set of open questions.

# 1 INTRODUCTION

Deep Learning has emerged as one of the cornerstones of large-scale machine learning. Deep Learning models are used for achieving state-of-the-art results on a wide variety of tasks including Computer Vision, Natural Language Processing and Reinforcement Learning; see (Bengio et al., 2016) and the references therein. The problem of training these networks is one of non-convex optimization. Mathematically, this can be represented as:

$$
\min  _ {x \in \mathbb {R} ^ {n}} f (x) := \frac {1}{M} \sum_ {i = 1} ^ {M} f _ {i} (x) \tag {1}
$$

where  $f_{i}$  is a loss function for data point  $i \in \{1,2,\dots ,M\}$  which captures the deviation of the model prediction from the data, and  $x$  is the vector of weights being optimized. The process of optimizing this function is also popularly called training of the network. Stochastic Gradient Descent (SGD) (Bottou, 1998; Sutskever et al., 2013) and its variants are often used for training deep networks. Generically, these methods minimize the objective function  $f$  by iteratively taking steps of the form:

$$
x _ {k + 1} = x _ {k} - \alpha_ {k} \left(\frac {1}{\left| B _ {k} \right|} \sum_ {i \in B _ {k}} \nabla f _ {i} \left(x _ {k}\right)\right), \tag {2}
$$

where  $B_{k}$  is the batch sampled from the data set and  $\alpha_{k}$  is the step size at iteration  $k$ . These methods can be interpreted as gradient descent using noisy gradients (Bottou, 1998), which and are often

referred to as mini-batch gradients with batch size  $|B_k|$ . SGD and its variants are employed in a small-batch regime, where  $|B_k| \ll M$  and typically  $|B_k| \in \{32, 64, \dots, 512\}$ . These configurations have been successfully used in practice for a large number of applications; see e.g. (Simonyan & Zisserman, 2014; Graves et al., 2013; Mnih et al., 2013). Many theoretical properties of these methods are known. These include guarantees of: (a) convergence to minimizers of strongly-convex functions and to stationary points for non-convex functions (Bottou et al., 2016), (b) saddle-point avoidance (Ge et al., 2015; Lee et al., 2016), and (c) robustness to input data (Hardt et al., 2015).

Stochastic gradient methods have, however, a major drawback: owing to the sequential nature of the iteration and small batch sizes, there is limited avenue for parallelization. While some efforts have been made to parallelize SGD for Deep Learning (Dean et al., 2012; Das et al., 2016; Zhang et al., 2015), the speed-ups and scalability obtained are often limited by the small batch sizes. One natural avenue for improving parallelism is to increase the batch size  $|B_k|$ . This increases the amount of computation per iteration, which can be effectively distributed. However, practitioners have observed that this leads to a significant loss in generalization performance; see e.g. (LeCun et al., 2012). In other words, the performance of the model on testing data sets is often worse when trained with large-batch methods as compared to small-batch methods. In our experiments, we have found the drop in generalization (also called generalization gap) to be as high as  $5\%$  even for smaller networks.

In this paper, we present numerical results that shed light into this drawback of large-batch methods. We observe that the generalization gap can be explained through the marked sharpness of the minimizers obtained by large-batch methods. This motivates efforts at remedying the generalization problem, as a training algorithm that employs large batches without sacrificing generalization performance would have the ability to scale to a much larger number of nodes than is possible today. This could potentially reduce the training time by orders-of-magnitude; we present an idealized performance model in the Appendix C to support this claim.

The paper is organized as follows. In the remainder of this section, we define the notation used in this paper, and in Section 2 we present our main findings and their supporting numerical evidence. In Section 3 we explore the performance of small-batch methods, and in Section 4 we briefly discuss the relationship between our results and recent theoretical work. We conclude with open questions concerning the explanation of the generalization gap and possible modifications to make large-batch training viable. In Appendix E, we present some attempts to overcome the problems of large-batch training.

# 1.1 NOTATION

We use the notation  $f_{i}$  to denote the composition of loss function and a prediction function corresponding to the  $i^{th}$  data point. The vector of weights is denoted by  $x$  and is subscripted by  $k$  to denote an iteration. We use the term small-batch (SB) method to denote SGD, or one of its variants like ADAM (Kingma & Ba, 2015) and ADAGRAD (Duchi et al., 2011), with the proviso that the gradient approximation is based on a small mini-batch. In our setup, the batch  $B_{k}$  is randomly sampled and its size is kept fixed for every iteration. We use the term large-batch (LB) method to denote any training algorithm that uses a large mini-batch. In our experiments ADAM will be used to illustrate the behavior of both a small or a large batch method.

# 2 DRAWBACKS OF LARGE-BATCH METHODS

# 2.1 OUR MAIN OBSERVATION

As mentioned in Section 1, practitioners have observed a generalization gap when using large-batch methods for training deep learning models. Interestingly, this is despite the fact that large-batch methods usually yield a similar value of the training function as small-batch methods. One may put forth the following as possible causes for this phenomenon: (i) LB methods over-fit the model relative to SB methods; (ii) LB methods are attracted to saddle points; (iii) LB methods lack the explorative properties of SB methods and tend to zoom-in on the minimizer closest to the initial point; (iv) SB and LB methods converge to qualitatively different minimizers with differing generalization properties. The data presented in this paper supports the last two conjectures.

The main observation of this paper is as follows:

The lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function. These minimizers are characterized by a significant number of large positive eigenvalues in  $\nabla^2 f(x)$  and tend to generalize less well. In contrast, small-batch methods converge to flat minimizers characterized by having numerous small eigenvalues of  $\nabla^2 f(x)$ . We have observed that the loss function landscape of deep neural networks is such that large-batch methods are almost invariably attracted to regions with sharp minimizers and that, unlike small-batch methods, are unable to escape basins of these minimizers.

The concept of sharp and flat minimizers have been discussed in the statistics and machine learning literature. (Hochreiter & Schmidhuber, 1997) (informally) define a flat minimizer  $\bar{x}$  as one for which the function varies slowly in a relatively large neighborhood of  $\bar{x}$ . In contrast, a sharp minimizer  $\hat{x}$  is such that the function increases rapidly in a small neighborhood of  $\hat{x}$ . A flat minimum can be described with low precision, whereas a sharp minimum requires high precision. The large sensitivity of the training function at a sharp minimizer negatively impacts the ability of the trained model to generalize on new data; see Figure 1 for an illustration. This can be explained through the lens of both the minimum description length (MDL) theory (Rissanen, 1983) and Bayesian learning (MacKay, 1992). Succinctly, the MDL theory states that statistical models that require fewer bits to describe (i.e., are of low complexity) generalize better. Since flat minimizers can be specified with lower precision as compared to sharp minimizers, they tend to have better generalization performance. A similar explanation is proffered through the Bayesian view of deep learning.

![](images/6b60c7ae9e5cbba03b53272c9f6882d0e43d246f7d5dc76689025e64800a7fcb.jpg)  
Figure 1: A Conceptual Sketch of Flat and Sharp Minima. The Y-axis indicates value of the loss function and the X-axis the variables (parameters)

# 2.2 NUMERICAL EXPERIMENTS

In this section, we present numerical results to justify the observations made above. To this end, we make use of the visualization technique employed by (Goodfellow et al., 2014b) and a proposed heuristic metric of sharpness (Equation (3)). We consider 6 multi-class classification network configurations for our experiments; they are described in Table 1. The details about the data sets and network configurations are presented in Appendices A and B respectively. As is common for such problems, we use the mean cross entropy loss as the objective function  $f$ .

The networks were chosen to exemplify popular configurations used in practice like AlexNet (Krizhevsky et al., 2012) and VGGNet (Simonyan & Zisserman, 2014). Results on other networks and using other initialization strategies, activation functions, and data sets showed similar behavior. Since the goal of our work is not to achieve state-of-the-art accuracy or time-to-solution on these tasks but rather to characterize the nature of the minima for LB and SB methods, we only describe the final testing accuracy in the main paper and ignore convergence trends.

For all experiments, we used  $10\%$  of the training data as batch size for the large-batch experiments and 256 data points for small-batch experiments. We used the ADAM optimizer for both regimes.

Table 1: Network Configurations  

<table><tr><td>Name</td><td>Network Type</td><td>Architecture</td><td>Data set</td></tr><tr><td>F1</td><td>Fully Connected</td><td>Section B.1</td><td>MNIST (LeCun et al., 1998a)</td></tr><tr><td>F2</td><td>Fully Connected</td><td>Section B.2</td><td>TIMIT (Garofolo et al., 1993)</td></tr><tr><td>C1</td><td>(Shallow) Convolutional</td><td>Section B.3</td><td>CIFAR-10 (Krizhevsky &amp; Hinton, 2009)</td></tr><tr><td>C2</td><td>(Deep) Convolutional</td><td>Section B.4</td><td>CIFAR-10</td></tr><tr><td>C3</td><td>(Shallow) Convolutional</td><td>Section B.3</td><td>CIFAR-100 (Krizhevsky &amp; Hinton, 2009)</td></tr><tr><td>C4</td><td>(Deep) Convolutional</td><td>Section B.4</td><td>CIFAR-100</td></tr></table>

Experiments with other optimizers for the large-batch experiments, including ADAGRAD (Duchi et al., 2011), SGD(Sutskever et al., 2013) and adaQN (Keskar & Berahas, 2016), led to similar results. All experiments were conducted 5 times from different (uniformly distributed random) starting points and we report both mean and standard-deviation of measured quantities. The baseline performance for our setup is presented Table 2. From this, we can observe that on all networks, both approaches led to high training accuracy but there is a significant difference in the generalization performance. The networks were trained, without any budget or limits, until the loss function ceased to improve.

Table 2: Performance of small-batch (SB) and large-batch (LB) variants of ADAM on the 6 networks listed in Table 1  

<table><tr><td rowspan="2">Name</td><td colspan="2">Training Accuracy</td><td colspan="2">Testing Accuracy</td></tr><tr><td>SB</td><td>LB</td><td>SB</td><td>LB</td></tr><tr><td>F1</td><td>99.66% ± 0.05%</td><td>99.92% ± 0.01%</td><td>98.03% ± 0.07%</td><td>97.81% ± 0.07%</td></tr><tr><td>F2</td><td>99.99% ± 0.03%</td><td>98.35% ± 2.08%</td><td>64.02% ± 0.2%</td><td>59.45% ± 1.05%</td></tr><tr><td>C1</td><td>99.89% ± 0.02%</td><td>99.66% ± 0.2%</td><td>80.04% ± 0.12%</td><td>77.26% ± 0.42%</td></tr><tr><td>C2</td><td>99.99% ± 0.04%</td><td>99.99% ± 0.01%</td><td>89.24% ± 0.12%</td><td>87.26% ± 0.07%</td></tr><tr><td>C3</td><td>99.56% ± 0.44%</td><td>99.88% ± 0.30%</td><td>49.58% ± 0.39%</td><td>46.45% ± 0.43%</td></tr><tr><td>C4</td><td>99.10% ± 1.23%</td><td>99.57% ± 1.84%</td><td>63.08% ± 0.5%</td><td>57.81% ± 0.17%</td></tr></table>

We emphasize that the generalization gap is not due to over-fitting or over-training as commonly observed in statistics. These phenomena manifest themselves in the form of a testing accuracy curve that, at a certain iterate peaks, and then decays due to the model learning idiosyncrasies of the training data. This is not what we observe in our experiments; see Figure 2 for the training-testing curve of the  $F_{2}$  and  $C_{1}$  networks, which are representative of the rest. As such, early-stopping heuristics aimed at preventing models from over-fitting would not help reduce the generalization gap. The difference between the training and testing accuracies for the networks is due to the specific choice of the network (e.g. AlexNet, VGGNet etc.) and is not the focus of this study. Rather, our goal is to study the source of the testing performance disparity of the two regimes, SB and LB, on a given network model.

![](images/d8c1a89c795676d5bf3e4b4bac4dff49f1665de51bce59df8a4a72540291cf37.jpg)  
(a) Network  $F_{2}$

![](images/86979cc9c9fd30c94022046e99ffff32fdc43b0fe050354ed6d2a07c96687f71.jpg)  
(b) Network  $C_1$  
Figure 2: Training and testing accuracy for SB and LB methods as a function of epochs

# 2.2.1 PARAMETRIC PLOTS

We first present parametric 1-D plots of the function as described in (Goodfellow et al., 2014b). Let  $x_{s}^{\star}$  and  $x_{\ell}^{\star}$  indicate the solutions obtained by running ADAM using small and large batch sizes respectively. We plot the loss function, on both training and testing data sets, along the line-segment containing the two points. Specifically, for  $\alpha \in [-1,2]$ , we plot the function  $f(\alpha x_{\ell}^{\star} + (1 - \alpha)x_{s}^{\star})$  and also superimpose the classification accuracy at the intermediate points; see Figure 3<sup>1</sup>. For this experiment, we randomly chose a pair of SB and LB minimizers from the 5 trials used to generate the data in Table 2. The plots show that the LB minima are strikingly sharper than the SB minima in this one-dimensional manifold. The plots in Figure 3 only explore a linear slice of the function, but in Figure 7 in Appendix D, we plot  $f(\sin (\frac{\alpha\pi}{2})x_{\ell}^{\star} + \cos (\frac{\alpha\pi}{2})x_{s}^{\star})$  to monitor the function along a curved path between the two minimizers. There too, the relative sharpness of the minima is evident.

# 2.2.2 SHARPNESS OF MINIMA

So far, we have used the term sharp minimizer loosely, but we noted that this concept has received attention in the literature (Hochreiter & Schmidhuber, 1997). For our purposes the sharpness of a minimizer can be best characterized by the magnitude of the eigenvalues of  $\nabla^2 f(x)$ , but given the prohibitive cost of this computation in deep learning applications, we propose a heuristic metric that, although imperfect, is computationally feasible, even for large networks. It is based on exploring a small neighborhood of a solution and computing the largest value that the function  $f$  can attain in that neighborhood. We use that value to measure the sensitivity of the training function at the given local minimizer. Now, if  $f$  attains a large value only in a small subspace of  $\mathbb{R}^n$ , our measure could be misleading. To verify that such a subspace is not small, we also choose random manifolds and compute the largest value of  $f$  in a small neighborhood of a minimizer restricted to those manifolds. For that purpose, we introduce an  $n \times p$  matrix  $A$ , whose columns are randomly generated. Here  $p$  denotes the dimension of the manifold, which in our experiments is chosen as  $p = 100$ .

Specifically, let  $\mathcal{C}_{\epsilon}$  denote the neighborhood around the solution that is explored and let  $A \in \mathbb{R}^{n \times p}$  denote a sub-sampling matrix whose column-space defines the subspace where the function will be explored. In order to ensure invariance of sharpness to problem dimension, sparsity and scale, we define the constraint set  $\mathcal{C}_{\epsilon}$  as:

$$
\mathcal {C} _ {\epsilon} = \{z \in \mathbb {R} ^ {p}: - \epsilon (| (A ^ {+} x) _ {i} | + 1) \leq z _ {i} \leq \epsilon (| (A ^ {+} x) _ {i} | + 1) \quad \forall i \in \{1, 2, \dots , p \} \},
$$

where  $A^{+}$  denotes the pseudo-inverse of  $A$ . We can now define our heuristic measure of sharpness.

Metric 2.1. Given  $x \in \mathbb{R}^n$ ,  $\epsilon > 0$  and  $A \in \mathbb{R}^{n \times p}$ , we define the  $(C_{\epsilon}, A)$ -sharpness of  $f$  at  $x$  as:

$$
\phi_ {x, f} (\epsilon , A) := \frac {\left(\max  _ {y \in \mathcal {C} _ {\epsilon}} f (x + A y)\right) - f (x)}{1 + f (x)} \times 1 0 0. \tag {3}
$$

Unless specified otherwise, we use this metric for sharpness for the rest of the paper; if  $A$  is not specified, it is assumed to be the identity matrix,  $I_{n}$ . (We note in passing that, in the convex optimization literature, the term sharp minimum has a different definition (Ferris, 1988), but that concept is not useful for our purposes.)

We present the values of the metric (3) for the minimizers of the various problems in Tables 3 and 4. The former explores the full-space (i.e.,  $A = I_{n}$ ) while the latter uses a randomly sampled  $n \times 100$  dimensional matrix  $A$ . We report results with two values of  $\epsilon$ ,  $(10^{-3}, 5 \cdot 10^{-4})$ . In all experiments, we solve the maximization problem in Equation (3) in exactly by applying 10 iterations of L-BFGS-B (Byrd et al., 1995). This limit on the number of iterations was necessitated by the large cost of evaluating the true objective  $f$ . Both tables show a 1-2 order-of-magnitude difference between the values of our metric for the SB and LB regimes. These results reinforce the view that the solutions obtained by a large-batch method defines points of larger sensitivity of the training function. In Appendix E, we describe possible approaches to remedy this generalization problem of LB methods. These approaches include data augmentation, conservative training and adversarial training. Our preliminary findings show that these approaches help reduce the generalization gap but still lead to relatively sharp minimizers and as such, do not completely remedy the problem.

![](images/c73699ee86c7dd16baf2bcc7fa261d9eae4ec9244f1e2583fd8f12d0adb5c05d.jpg)  
(a)  $F_{1}$

![](images/c9ba938f698a1dc3ff33efd23ec7e584f17d07f10528d94c5b88c901c5eefd2f.jpg)  
(b)  $F_{2}$

![](images/c8818e56c3f35ba5b961b0427ed5514a14d3a79c38d5ccd40429ac8c73fe5472.jpg)  
(c)  $C_1$

![](images/be8244886fb59502fbf9d66817f4917909764ca040bec28198ca11f35a479652.jpg)  
(d)  $C_2$

![](images/d9a462a2761d6c583c63d1c315d31973739f93559236ecd951eabc275bad0836.jpg)  
(e)  $C_3$

![](images/6f8ba2fd53b02e3e08bb6831fcf03fc0c8d9af3254ad1f74d8b750221366469e.jpg)  
(f)  $C_4$  
Figure 3: Parametric Plots – Linear (Left vertical axis corresponds to cross-entropy loss,  $f$ , and right vertical axis corresponds to classification accuracy; solid line indicates training data set and dashed line indicated testing data set);  $\alpha = 0$  corresponds to the SB minimizer while  $\alpha = 1$  corresponds to the LB minimizer

Notice that the proposed heuristic metric is closely related to the spectrum of  $\nabla^2 f(x)$ . Assuming  $\epsilon$  to be small enough, when  $A = I_n$ , the value of the metric relates to the largest eigenvector of the  $\nabla^2 f(x)$  while in the case when  $A$  is randomly sampled, the value relates to the Ritz value of  $\nabla^2 f(x)$  projected onto the column-space of  $A$ .

# 3 SUCCESS OF SMALL-BATCH METHODS

It is often reported that when increasing the batch size for a problem, there exists a threshold after which there is deterioration in the quality of the model. This behavior can be observed for the  $F_{2}$  and  $C_{1}$  networks in Figure 4. In both of these experiments, there is a batch size ( $\approx 15000$  for  $F_{2}$  and  $\approx 500$  for  $C_{1}$ ) after which there is a large drop in testing accuracy. Notice also that the upward drift

Table 3: Sharpness of Minima in Full Space  

<table><tr><td rowspan="2"></td><td colspan="2">ε=10-3</td><td colspan="2">ε=5·10-4</td></tr><tr><td>SB</td><td>LB</td><td>SB</td><td>LB</td></tr><tr><td>F1</td><td>1.23 ± 0.83</td><td>205.14 ± 69.52</td><td>0.61 ± 0.27</td><td>42.90 ± 17.14</td></tr><tr><td>F2</td><td>1.39 ± 0.02</td><td>310.64 ± 38.46</td><td>0.90 ± 0.05</td><td>93.15 ± 6.81</td></tr><tr><td>C1</td><td>28.58 ± 3.13</td><td>707.23 ± 43.04</td><td>7.08 ± 0.88</td><td>227.31 ± 23.23</td></tr><tr><td>C2</td><td>8.68 ± 1.32</td><td>925.32 ± 38.29</td><td>2.07 ± 0.86</td><td>175.31 ± 18.28</td></tr><tr><td>C3</td><td>29.85 ± 5.98</td><td>258.75 ± 8.96</td><td>8.56 ± 0.99</td><td>105.11 ± 13.22</td></tr><tr><td>C4</td><td>12.83 ± 3.84</td><td>421.84 ± 36.97</td><td>4.07 ± 0.87</td><td>109.35 ± 16.57</td></tr></table>

Table 4: Sharpness of Minima in Random Subspaces of Dimension 100  

<table><tr><td></td><td colspan="2">ε = 10-3</td><td colspan="2">ε = 5 · 10-4</td></tr><tr><td></td><td>SB</td><td>LB</td><td>SB</td><td>LB</td></tr><tr><td>F1</td><td>0.11 ± 0.00</td><td>9.22 ± 0.56</td><td>0.05 ± 0.00</td><td>9.17 ± 0.14</td></tr><tr><td>F2</td><td>0.29 ± 0.02</td><td>23.63 ± 0.54</td><td>0.05 ± 0.00</td><td>6.28 ± 0.19</td></tr><tr><td>C1</td><td>2.18 ± 0.23</td><td>137.25 ± 21.60</td><td>0.71 ± 0.15</td><td>29.50 ± 7.48</td></tr><tr><td>C2</td><td>0.95 ± 0.34</td><td>25.09 ± 2.61</td><td>0.31 ± 0.08</td><td>5.82 ± 0.52</td></tr><tr><td>C3</td><td>17.02 ± 2.20</td><td>236.03 ± 31.26</td><td>4.03 ± 1.45</td><td>86.96 ± 27.39</td></tr><tr><td>C4</td><td>6.05 ± 1.13</td><td>72.99 ± 10.96</td><td>1.89 ± 0.33</td><td>19.85 ± 4.12</td></tr></table>

in value of the sharpness is considerably reduced around this threshold. Similar thresholds exist for the other networks in Table 1.

Let us now consider the behavior of SB methods, which use noisy gradients in the step computation. From the results reported in the previous section, it appears that noise in the gradient pushes the iterates out of the basin of attraction of sharp minimizers and encourages movement towards a flatter minimizer where noise will not cause exit from that basin. When the batch size is greater than the threshold mentioned above, the noise in the stochastic gradient is not sufficient to cause ejection from the initial basin leading to convergence to sharper a minimizer.

To explore that in more detail, consider the following experiment. We train the network for 100 epochs using ADAM with a batch size of 256, and retain the iterate after each epoch in memory. Using these 100 iterates as starting points we train the network using a LB method for 100 epochs and receive a 100 piggybacked (or warm-started) large-batch solutions. We plot in Figure 5 the testing accuracy and sharpness of these large-batch solutions, along with the testing accuracy of the small-batch iterates. Note that when warm-started with only a few initial epochs, the LB method does not yield a generalization improvement. The concomitant sharpness of the iterates also stays high. It appears that this is the region wherein the noise is crucial to avoid sharp minima. However,

![](images/fdf51b39aa5dc645492ff7b6dc20f2545fcc0ed9e6078d507066f3645c39c90c.jpg)  
(a)  $F_{2}$

![](images/be6764223fdf6e1dfebb41f3cfce0cd47a3a4b676548f8e3f74da6177be5f09a.jpg)  
(b)  $C_1$  
Figure 4: Testing Accuracy and Sharpness v/s Batch Size. The X-axis corresponds to the batch size used for training the network for 100 epochs, left Y-axis corresponds to the testing accuracy at the final iterate and right Y-axis corresponds to the sharpness of that iterate.

![](images/1f899f5a2fb80cef44e7a53e5f14bc07695d9e6ff0dbf33720bbca10ce3e7d74.jpg)  
(a)  $F_{2}$

![](images/0da9d89cbcb81b92783ca8ad41767580e426e4d18e5f56bfcdd78021a4b7964e.jpg)  
(b)  $C_1$

![](images/cb83a1599bf69e86fd6c0bf9547c5b6d296b98c1144bcf167097f138de4bc8d4.jpg)  
Figure 5: Warm-starting experiments. The upper figures report the testing accuracy of the SB method (blue line) and the testing accuracy of the warm started (piggybacked) LB method (red line), as a function of the number of epochs of the SB method. The lower figures plot the sharpness measure (3) for the solutions obtained by the piggybacked LB method v/s the number of warm-starting epochs of the SB method.  
(a)  $F_{2}$

![](images/d42b65c1009faa7441280fa9368e6b39b2bad252e39c2412d6bb61bbd260ce2b.jpg)  
(b)  $C_1$  
Figure 6: Sharpness v/s Cross Entropy Loss for SB and LB methods.

after certain number of epochs of warm-starting, the accuracy improves and sharpness of the large-batch iterates drop. This happens, apparently, when the SB method has ended its exploration phase and discovered a flat minimizer; the large-batch method is then able to converge towards it, leading to good testing accuracy.

It has been speculated that LB methods tend to be attracted to minimizers close to the starting point, whereas SB methods move away and locate minimizers that are farther away. Our numerical experiments support this view: we observed that the ratio of  $\| x_s^\star -x_0\| _2$  and  $\| x_{\ell}^{\star} - x_{0}\|_{2}$  was in the range of 3-10.

In order to further illustrate the qualitative difference between the solutions obtained by SB and LB methods, we plot in Figure 6 our sharpness measure against the loss function (cross entropy) for one random trial of the  $F_{2}$  and  $C_{1}$  networks. For larger values of the loss function, i.e., near the initial point, SB and LB method yield similar values of sharpness. As the loss function reduces, the sharpness of the iterates corresponding to the LB method rapidly increases, whereas for the SB method the sharpness stays relatively constant initially and then reduces, suggesting an exploration phase followed by convergence to a flat minimizer.

# 4 DISCUSSION AND CONCLUSION

In this paper, we present numerical experiments that support the view that the presence of (and convergence to) sharp minimizers is the cause for the poor generalization of large-batch methods for

deep learning. To this end, we provide one-dimensional parametric plots and perturbation measures for a variety of deep learning architectures. In Appendix E, we describe our attempts to remedy the problem, including data augmentation, conservative training and robust optimization. Our preliminary investigation suggests that these strategies do not completely correct the problem; they improve the generalization of large-batch methods but still lead to relatively sharp minima. Another prospective remedy includes the use of dynamic sampling where the batch size is increased gradually as the iteration progresses (Byrd et al., 2012; Friedlander & Schmidt, 2012). The potential viability of this approach is suggested by our warm-starting experiments (see Figure 5) wherein high testing accuracy is achieved using a large-batch method that is warm-start with a small-batch method.

Recently, a number of researchers have described interesting theoretical properties of the loss surface of deep neural networks; see e.g. (Choromanska et al., 2015; Soudry & Carmon, 2016; Lee et al., 2016). Their work shows that, under certain regularity assumptions, the loss function of deep learning models is fraught with many local minimizers and that many of these minimizers correspond to a similar loss function value. Our results are in alignment these observations since, in our experiments, both sharp and flat minimizers have very similar loss function values. We do not know, however, if the theoretical models mentioned above provide information about the existence and density of sharp minimizers of the loss surface.

Our results suggest some questions: (a) can one prove that large-batch methods typically converge to sharp minimizers of deep learning training functions? (In this paper, we only provided some numerical evidence.); (b) what is the relative density of the two kinds of minima?; (c) can one design neural network architectures for various tasks that are suitable to the properties of LB methods?; (d) can the networks be initialized in a way that enables LB methods to succeed?; (e) is it possible, through algorithmic or regulatory means to steer LB methods away from sharp minimizers?

# REFERENCES

Yoshua Bengio, Ian Goodfellow, and Aaron Courville. Deep learning. Book in preparation for MIT Press, 2016. URL http://www.deeplearningbook.org.  
Dimitris Bertsimas, Omid Nohadani, and Kwong Meng Teo. Robust optimization for unconstrained simulation-based problems. Operations Research, 58(1):161-178, 2010.  
Léon Bottou. Online learning and stochastic approximations. On-line learning in neural networks, 17(9):142, 1998.  
Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. arXiv preprint arXiv:1606.04838, 2016.  
Richard H Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing, 16(5):1190-1208, 1995.  
Richard H Byrd, Gillian M Chin, Jorge Nocedal, and Yuchen Wu. Sample size selection in optimization methods for machine learning. Mathematical programming, 134(1):127-155, 2012.  
Anna Choromanska, Mikael Henaff, Michael Mathieu, Gérard Ben Arous, and Yann LeCun. The loss surfaces of multilayer networks. In AISTATS, 2015.  
Dipankar Das, Sasikanth Avancha, Dheevatsa Mudigere, Karthikeyan Vaidynathan, Srinivas Sridharan, Dhiraj Kalamkar, Bharat Kaul, and Pradeep Dubey. Distributed deep learning using synchronous stochastic gradient descent. arXiv preprint arXiv:1602.06709, 2016.  
Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior, Paul Tucker, Ke Yang, Quoc V Le, et al. Large scale distributed deep networks. In Advances in neural information processing systems, pp. 1223-1231, 2012.  
J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. The Journal of Machine Learning Research, 12:2121-2159, 2011.  
Michael Charles Ferris. Weak sharp minima and penalty functions in mathematical programming. PhD thesis, University of Cambridge, 1988.

Michael P Friedlander and Mark Schmidt. Hybrid deterministic-stochastic methods for data fitting. SIAM Journal on Scientific Computing, 34(3):A1380-A1405, 2012.  
John S Garofolo, Lori F Lamel, William M Fisher, Jonathan G Fiscus, David S Pallett, Nancy L Dahlgren, and Victor Zue. Timit acoustic-phonetic continuous speech corpus. Linguistic data consortium, Philadelphia, 33, 1993.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle pointsonline stochastic gradient for tensor decomposition. In Proceedings of The 28th Conference on Learning Theory, pp. 797-842, 2015.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014a.  
Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. arXiv preprint arXiv:1412.6544, 2014b.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In 2013 IEEE international conference on acoustics, speech and signal processing, pp. 6645–6649. IEEE, 2013.  
M. Hardt, B. Recht, and Y. Singer. Train faster, generalize better: Stability of stochastic gradient descent. arXiv preprint arXiv:1509.01240, 2015.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1-42, 1997.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Nitish Shirish Keskar and Albert S. Berahas. _adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs_, pp. 1-16. Springer International Publishing, Cham, 2016.  
D. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR 2015), 2015.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998a.  
Yann LeCun, Corinna Cortes, and Christopher JC Burges. The mnist database of handwritten digits, 1998b.  
Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pp. 9-48. Springer, 2012.  
Jason D Lee, Max Simchowitz, Michael I Jordan, and Benjamin Recht. Gradient descent converges to minimizers. University of California, Berkeley, 1050:16, 2016.  
Mu Li, Tong Zhang, Yuqiang Chen, and Alexander J Smola. Efficient mini-batch training for stochastic optimization. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 661-670. ACM, 2014.  
David JC MacKay. A practical bayesian framework for backpropagation networks. Neural computation, 4(3):448-472, 1992.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

Hossein Mobahi. Training recurrent neural networks by diffusion. arXiv preprint arXiv:1601.04114, 2016.  
Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Lukas Burget, Ondrej Glembek, Nagendra Goel, Mirko Hannemann, Petr Motlicek, Yanmin Qian, Petr Schwarz, et al. The kaldi speech recognition toolkit. In IEEE 2011 workshop on automatic speech recognition and understanding, number EPFL-CONF-192584. IEEE Signal Processing Society, 2011.  
Jorma Rissanen. A universal prior for integers and estimation by minimum description length. The Annals of statistics, pp. 416-431, 1983.  
Uri Shaham, Yutaro Yamada, and Sahand Negahban. Understanding adversarial training: Increasing local stability of neural nets through robust optimization. arXiv preprint arXiv:1511.05432, 2015.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Daniel Soudry and Yair Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. arXiv preprint arXiv:1605.08361, 2016.  
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning (ICML 2013), pp. 1139-1147, 2013.  
Sixin Zhang, Anna E Choromanska, and Yann LeCun. Deep learning with elastic averaging sgd. In Advances in Neural Information Processing Systems, pp. 685-693, 2015.  
Stephan Zheng, Yang Song, Thomas Leung, and Ian Goodfellow. Improving the robustness of deep neural networks via stability training. arXiv preprint arXiv:1604.04326, 2016.
