# SEMANTIC NOISE MODELING FOR BETTER REPRESENTATION LEARNING

Hyo-Eun Kim* and Sangheum Hwang

Lunit Inc.

Seoul, South Korea

{hekim, shwang}@lunit.io

Kyunghyun Cho

Courant Institute of Mathematical Sciences and Centre for Data Science

New York University

New York, NY 10012, USA

kyunghyun.cho@nyu.edu

# ABSTRACT

Latent representation learned from multi-layered neural networks via hierarchical feature abstraction enables recent success of deep learning. Under the deep learning framework, generalization performance highly depends on the learned latent representation which is obtained from an appropriate training scenario with a task-specific objective on a designed network model. In this work, we propose a novel latent space modeling method to learn better latent representation. We designed a neural network model based on the assumption that good base representation can be attained by maximizing the total correlation between the input, latent, and output variables. From the base model, we introduce a semantic noise modeling method which enables class-conditional perturbation on latent space to enhance the representational power of learned latent feature. During training, latent vector representation can be stochastically perturbed by a modeled class-conditional additive noise while maintaining its original semantic feature. It implicitly brings the effect of semantic augmentation on the latent space. The proposed model can be easily learned by back-propagation with common gradient-based optimization algorithms. Experimental results show that the proposed method helps to achieve performance benefits against various previous approaches. We also provide the empirical analyses for the proposed class-conditional perturbation process including t-SNE visualization.

# 1 INTRODUCTION

Enhancing the generalization performance against unseen data given some sample data is the main objective in machine learning. Under that point of view, deep learning has been achieved many breakthroughs in several domains such as computer vision (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016), natural language processing (Collobert & Weston, 2008; Bahdanau et al., 2015), and speech recognition (Hinton et al., 2012; Graves et al., 2013). Deep learning is basically realized on deep layered neural network architecture, and it learns appropriate task-specific latent representation based on given training data. Better latent representation learned from training data results in better generalization over the future unseen data. Representation learning or latent space modeling becomes one of the key research topics in deep learning. During the past decade, researchers focused on unsupervised representation learning and achieved several remarkable landmarks on deep learning history (Vincent et al., 2010; Hinton et al., 2006; Salakhutdinov & Hinton, 2009). In terms of utilizing good base features for supervised learning, the base representation learned from unsupervised learning can be a good solution for supervised tasks (Bengio et al., 2007; Masci et al., 2011).

![](images/10ea11895b7fca6fdf0eb9bbab381c7b3b1b4237f0e4deeda917011323d8021f.jpg)

![](images/73e5f68161116aa5a45310b1384292a3ea7b6ac4499052febf850332b572bba8.jpg)

![](images/99a517e2fc827dc463b645835ba3114ac862b90f3f15a84a514a480374abb0b7.jpg)  
Figure 1: (a) Standard feed-forward neural network model, (b) feed-forward neural network model with reconstruction paths, and (c) feed-forward neural network model with reconstruction and stochastic perturbation paths.

The definition of 'good' representation is, however, different according to target tasks. In unsupervised learning, a model is learned from unlabelled examples. Its main objective is to build a model to estimate true data distribution given examples available for training, so the learned latent representation normally includes broadly-informative components of the raw input data (e.g., mutual information between the input and the latent variable can be maximized for this objective). In supervised learning, however, a model is learned from labelled examples. In the case of classification, a supervised model learns to discriminate input data in terms of the target task using corresponding labels. Latent representation is therefore obtained to maximize the performance on the target supervised tasks.

Since the meaning of good representations vary according to target tasks (unsupervised or supervised), pre-trained features from the unsupervised model are not be guaranteed to be useful for subsequent supervised tasks. Instead of the two stage learning strategy (unsupervised pre-training followed by supervised fine-tuning), several works focused on a joint learning model which optimizes unsupervised and supervised objectives concurrently, resulting in better generalization performance (Goodfellow et al., 2013; Larochelle & Bengio, 2008; Rasmus et al., 2015; Zhao et al., 2015; Zhang et al., 2016; Cho & Chen, 2014).

In this work, we propose a novel latent space modeling method for supervised learning. We define a good latent representation of standard feed-forward neural networks under the basis of information theory. Then, we introduce a semantic noise modeling method in order to enhance the generalization performance. The proposed method stochastically perturbs the latent representation of a training sample by injecting class-conditional additive noise. Since the additive noise is randomly sampled from a pre-defined probability distribution every training iteration, different latent vectors from a single training example can be used for training. The multiple different latent vectors produced from a single training example are semantically similar under the proposed class-conditional perturbation process, so we can expect semantic augmentation effect on the latent space.

Experiments are performed on two datasets; MNIST and CIFAR-10. The proposed model results in better classification performance compared to previous approaches through notable generalization effect (class-conditionally perturbed training samples well cover the distribution of unseen data).

# 2 METHODOLOGY

In a traditional feed-forward neural network model (Figure 1(a)), output  $Y$  of input data  $X$  is compared with its true label, and the error is propagated backward from top to bottom, which implicitly learns a task-specific latent representation  $Z$  of the input  $X$ . We assume that good latent representation  $Z$  is attained by maximizing the dependency among a set of random variables  $X, Y$ , and  $Z$ , which is known as total correlation or multiinformation (Watanabe, 1960). Note that the total correlation is equal to the sum of all pairwise mutual informations. The total correlation  $\mathbf{C}(X,Y,Z)$

for given random variables  $X$ ,  $Y$ , and  $Z$  under the condition  $P(X, Y, Z) = P(Y|Z)P(Z|X)P(X)$  from the relationship between the random variables (in Figure 1(a)) can be reduced to:

$$
\begin{array}{l} \mathbf {C} (X, Y, Z) = \mathbf {H} (X) + \mathbf {H} (Y) + \mathbf {H} (Z) - \mathbf {H} (X, Y, Z) \\ = \mathbf {H} (X) + \mathbf {H} (Y) + \mathbf {H} (Z) - (\mathbf {H} (Y | Z) + \mathbf {H} (Z | X) + \mathbf {H} (X)) \\ = \mathbf {I} (X; Z) + \mathbf {I} (Z; Y) \tag {1} \\ = \mathbf {H} (X) - \mathbf {H} (X | Z) + \mathbf {H} (Z) - \mathbf {H} (Z | Y) \\ \end{array}
$$

where  $\mathbf{I}(A;B)$  is the mutual information between random variables  $A$  and  $B$ , and  $\mathbf{H}(A)$  is the entropy of a random variable  $A$ . Our objective is to find the model parameter  $\theta$  which maximizes  $\mathbf{C}(X,Y,Z)$ . Since  $\mathbf{H}(X)$  and  $\mathbf{H}(Z)$  are non-negative, and  $\mathbf{H}(X)$  is constant in this case, the lower bound on  $\mathbf{C}(X,Y,Z)$  can be summarized as:

$$
\mathbf {C} (X, Y, Z) \geq - \mathbf {H} (X | Z) - \mathbf {H} (Z | Y). \tag {2}
$$

It is known that maximizing  $-\mathbf{H}(X|Z)$  can be formulated as minimizing the reconstruction error between the input  $x$  (sampled from  $X$ ) and its reconstruction  $x_{R}$  under the general audio-encoder framework (Vincent et al., 2010). Similarly, maximizing  $-\mathbf{H}(Z|Y)$  can be reformulated by minimizing the reconstruction error between  $z$  and its reconstruction  $z_{R}$ . The target objective can then be defined as follows:

$$
\min  _ {\theta} \lambda_ {1} L _ {r e c} (x, x _ {R}) + \lambda_ {2} L _ {r e c} (z, z _ {R}) \tag {3}
$$

where  $\theta$  and  $\lambda_{1,2}$  are model parameters to be learned and constant coefficients, and  $L_{rec}$  is a reconstruction loss.

Given an input sample  $x$ , feed-forwarded vectors and their reconstructions are attained deterministically by:

$$
z = f _ {\theta_ {1}} (x)
$$

$$
\begin{array}{l} y = f _ {\theta_ {2}} \left(f _ {\theta_ {1}} (x)\right) \\ x _ {R} = g _ {\theta_ {1} ^ {\prime}} (z) = g _ {\theta_ {1} ^ {\prime}} \left(f _ {\theta_ {1}} (x)\right) \end{array} \tag {4}
$$

$$
z _ {R} = g _ {\theta_ {2} ^ {\prime}} (y) = g _ {\theta_ {2} ^ {\prime}} \big (f _ {\theta_ {2}} \big (f _ {\theta_ {1}} (x) \big)
$$

where  $x_{R}$  and  $z_{R}$  are the reconstruction of  $x$  and  $z$  as shown in Figure 1(b).

For supervised learning, given a set of training pairs  $(x,t)$  where  $x$  and  $t$  are the input sample and its label, target objective under the model described in Figure 1(b) can be defined as below (with real-valued input samples, L2 loss  $L_{L2}$  is a proper choice for the reconstruction loss  $L_{rec}$ ):

$$
\min  _ {\theta : \left\{\theta_ {1}, \theta_ {1} ^ {\prime}, \theta_ {2}, \theta_ {2} ^ {\prime} \right\}} \lambda_ {1} L _ {L 2} (x, x _ {R}) + \lambda_ {2} L _ {L 2} (z, z _ {R}) + \lambda_ {3} L _ {N L L} (y, t) \tag {5}
$$

where  $L_{NLL}$  and  $\lambda_3$  are a negative log-likelihood loss for the target supervised task and a relative weighting factor for  $L_{NLL}$ , respectively. Note that Eq. (5) represents the 'proposed-base' in our experiment (see Section 4.3).

Based on the architecture shown in Figure 1(b) with the target objective in Eq. (5), we conjecture that stochastic perturbation on the latent space during training helps to achieve better generalization performance for supervised tasks. Figure 1(c) shows this strategy which integrates the stochastic perturbation process during training. Suppose that  $Z_{P}$  is a perturbed version of  $Z$ , and  $Y_{P}$  is an output which is feed-forwarded from  $Z_{P}$ . Given an input sample  $x$ ,

$$
z ^ {\prime} = z + z _ {e} \text {a n d} \hat {y} = f _ {\theta_ {2}} \left(z ^ {\prime}\right) \tag {6}
$$

![](images/3f34504e5d0b043b7e7ce68d4e201be12c47291990c2356de90a2985a87666a1.jpg)  
(a)

![](images/5d1a1531573100fbe21ce59b45e673be4fb56849b054b7b19a00184c9332cb98.jpg)  
(b)  
Figure 2: Previous works for supervised learning; (a) traditional feed-forward model, and (b) joint learning model with both supervised and unsupervised losses.

where  $z'$  and  $\hat{y}$  are samples of  $Z_P$  and  $Y_P$  respectively, and  $z_e$  is an additive noise used in the perturbation process of  $z$ . Based on the architecture shown in Figure 1(c), target objective can be modified as:

$$
\min  _ {\theta : \left\{\theta_ {1}, \theta_ {1} ^ {\prime}, \theta_ {2}, \theta_ {2} ^ {\prime} \right\}} \lambda_ {1} L _ {L 2} (x, x _ {R}) + \lambda_ {2} L _ {L 2} (z, z _ {R}) + \lambda_ {3} L _ {N L L} (y, t) + L _ {N L L} (\hat {y}, t). \tag {7}
$$

Direct random additive noise is not appropriate for  $z_{e}$  ('proposed-perturb (random)' in Section 4.3), since random perturbation might destroy the semantic feature of the original latent representation  $z$ . In order to maintain the semantics of the original latent representation after perturbation, we design a class-conditional additive noise which can be modeled based on the architecture described in Figure 1(b). We assume that the probability density function  $P(Y_{(j)}|X)$  is approximately Gaussian with the deterministic feed-forwardsed value  $y_{(j)}$  as a mean as below:

$$
P \left(Y _ {(j)} \mid X\right) = \mathcal {N} \left(f _ {\theta_ {2}} \left(f _ {\theta_ {1}} (x)\right) _ {(j)}, \sigma_ {(j)} ^ {2}\right) = y _ {(j)} + \mathcal {N} \left(0, \sigma_ {(j)} ^ {2}\right) \tag {8}
$$

where  $Y_{(j)}$  and  $\sigma_{(j)}$  are the  $j$ -th element of the random vector  $Y$  and a standard deviation for  $Y_{(j)}$ . Now, the class-conditionally perturbed  $z$  (i.e.  $z'$  in Eq. (6)) can be reconstructed from the class-conditionally perturbed  $y$  (i.e.  $y'$ ) through the decoding path  $g_{\theta_2'}$ . The semantic-preserving variation of  $y$  (i.e.  $y'$ ) can be modeled according to Eq. (8) by  $y' = y + y_e$ , where  $y_e$  is a random noise vector which is stochastically sampled from the Gaussian distribution. From  $y'$ , class-conditional additive noise on the latent space,  $z_e$  ('proposed-perturb (class-conditional)' in Section 4.3), can be approximately modeled as below:

$$
z _ {R} = g _ {\theta_ {2} ^ {\prime}} (y)
$$

$$
z _ {R} ^ {\prime} = g _ {\theta_ {2} ^ {\prime}} (y ^ {\prime}) = g _ {\theta_ {2} ^ {\prime}} (y + y _ {e}) \tag {9}
$$

$$
z _ {e} \simeq z _ {R} ^ {\prime} - z _ {R} = g _ {\theta_ {2} ^ {\prime}} (y + y _ {e}) - g _ {\theta_ {2} ^ {\prime}} (y).
$$

From the described semantic noise modeling process, we expect to achieve better representation on the latent space. The effect of the proposed model in terms of learned latent representation will be explained in more detail in Section 4.4.

# 3 RELATED WORKS

Previous works on deep neural networks for supervised learning can be categorized into two types as shown in Figure 2; (a) a general feed-forward neural network model (LeCun et al., 1998; Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016), and (b) a joint learning model which optimizes unsupervised and supervised objectives at the same time (Zhao et al., 2015; Zhang et al., 2016; Cho & Chen, 2014). Here are the corresponding objective functions:

$$
\min  _ {\theta : \left\{\theta_ {1}, \theta_ {2} \right\}} L _ {N L L} (y, t) \tag {10}
$$

$$
\min  _ {\theta : \left\{\theta_ {1}, \theta_ {1} ^ {\prime}, \theta_ {2} \right\}} \lambda L _ {L 2} \left(x, x _ {R}\right) + L _ {N L L} (y, t) \tag {11}
$$

![](images/93b9f2a4e38b891935b66303e72accfa880543af582f27d20edc25af5a2c7010.jpg)  
Figure 3: Ladder network; a representative model for semi-supervised learning (Rasmus et al., 2015).

where  $\lambda$  is a loss weighting factor between unsupervised and supervised losses.

Since the feed-forward neural network model is normally implemented with multiple layers in a deep learning framework, the joint learning model can be sub-classified into two types according to the type of reconstruction; reconstruction only with the input data  $x$  (Eq. (11)) and reconstruction with all the intermediate features including the input data  $x$  as follows:

$$
\min  _ {\theta} \lambda_ {0} L _ {L 2} (x, x _ {R}) + \sum_ {i} \lambda_ {i} L _ {L 2} \left(h _ {i}, h _ {i _ {R}}\right) + L _ {N L L} (y, t). \tag {12}
$$

where  $h_i$  and  $h_{i_R}$  are the  $i$ -th hidden representation and its reconstruction.

Another type of the joint learning model, a ladder network (Figure 3), was introduced for semi-supervised learning (Rasmus et al., 2015). The key concept of the ladder network is to obtain robust features by learning de-noising functions  $(g_{\theta'})$  of the representations at every layer of the model via reconstruction losses, and the supervised loss is combined with the reconstruction losses in order to build the semi-supervised model. The ladder network achieved the best performance in semi-supervised tasks, but it is not appropriate for supervised tasks especially with small-scale training samples (experimental analysis for supervised learning on MNIST is briefly summarized in Appendix (A2)). The proposed model in this work can be extended to semi-supervised learning, but our main focus is to enhance the representational power on latent space given labelled data for supervised learning. We leave the study for semi-supervised learning scenario based on the proposed methodology as our future research.

# 4 EXPERIMENTS

For quantitative analysis, we compare the proposed methodology with previous approaches described in Section 3; a traditional feed-forward supervised learning model and a joint learning model with two different types of reconstruction losses (reconstruction only with the first layer or with all the intermediate layers including the first layer). The proposed methodology includes a baseline model in Figure 1(b) as well as a stochastic perturbation model in Figure 1(c). Especially in the stochastic perturbation model, we compare the random and class-conditional perturbations and present some qualitative analysis on the meaning of the proposed perturbation methodology.

# 4.1 DATASETS

We experiment with two public datasets; MNIST and CIFAR-10. MNIST (10 classes) consists of 50k, 10k, and 10k  $28 \times 28$  gray-scale images for training, validation, and test datasets, respectively. CIFAR-10 (10 classes) consists of 50k and 10k  $32 \times 32$  3-channel images for training and test sets, respectively. We split the 50k CIFAR-10 training images into 40k and 10k for training and validation. Experiments are performed with different sizes of training set (from 10 examples per class to the entire training set) in order to verify the effectiveness of the proposed model in terms of generalization performance under varying sizes of training set.

![](images/fd4f44c3b0e606e457d47e5d052e8cae5da1f76ff297203f0a18318848757316.jpg)  
Figure 4: Target network architecture; 3 convolution and 2 fully-connected layers were used for MNIST, and 4 convolution and 3 fully-connected layers were used for CIFAR-10.

# 4.2 IMPLEMENTATION

Figure 4 shows the architecture of the neural network model used in this experiment.  $W$ 's are convolution or fully-connected weights (biases are excluded for visual brevity). Three convolution  $(3 \times 3$  (2) 32,  $3 \times 3$  (2) 64,  $3 \times 3$  (2) 96, where each item means the filter kernel size and (stride) with the number of filters) and two fully-connected (the numbers of output nodes are 128 and 10, respectively) layers are used for MNIST. Four convolution  $(5 \times 5$  (1) 64,  $3 \times 3$  (2) 64,  $3 \times 3$  (2) 64, and  $3 \times 3$  (2) 96) and three fully-connected (128, 128, and 10 nodes) layers are used for CIFAR-10. Weights on the decoding (reconstruction) path are tied with corresponding weights on the encoding path as shown in Figure 4.

In Figure 4,  $z'$  is perturbed directly from  $z$  by adding Gaussian random noise for random perturbation. For class-conditional perturbation,  $z'$  is indirectly generated from  $y'$  which is perturbed by adding random noise on  $y$  based on Eq. (9). For perturbation, base activation vector ( $z$  is the base vector for random perturbation and  $y$  is the base vector for class-conditional perturbation) is scaled to [0.0, 1.0], and the zero-mean Gaussian noise with 0.2 of standard deviation is added (via element-wise addition) on the normalized base activation. This perturbed scaled activation is de-scaled with the original min and max activations of the base vector.

Initial learning rates are 0.005 and 0.002 for MNIST and CIFAR-10, respectively. The learning rates are decayed by a factor of 5 every 40 epochs until the 120-th epoch. For both datasets, the minibatch size is set to 100, and the target objective is optimized using Adam optimizer (Kingma & Ba, 2015) with a momentum 0.9. All the  $\lambda$ 's for reconstruction losses in Eq. (11) and Eq. (12) are 0.03 and 0.01 for MNIST and CIFAR-10, respectively. The same weighting factors for reconstruction losses (0.03 for MNIST and 0.01 for CIFAR-10) are used for  $\lambda_1$  and  $\lambda_2$  in Eq (7), and 1.0 is used for  $\lambda_3$ .

Input data is first scaled to [0.0, 1.0] and then whitened by the average across all the training examples. In CIFAR-10, random cropping (24×24 image is randomly cropped from the original 32×32 image) and random horizontal flipping (mirroring) are used for data augmentation. We selected the network that performed best on the validation dataset for evaluation on the test dataset. All the experiments are performed with TensorFlow (Abadi et al., 2015).

# 4.3 QUANTITATIVE ANALYSIS

Table 1 shows the classification performance of previous approaches and the proposed methods. Three previous approaches (a traditional feed-forward model, a joint learning model with the input reconstruction loss, and a joint learning model with reconstruction losses of all the intermediate layers including the input layer) are compared with three proposed methods (the baseline model in Figure 1(b), and the stochastic perturbation model in Figure 1(c) with two different perturbation methods; random and class-conditional).

As we expected, maximizing the total correlation (proposed-base) learns better latent representation, and the model with the class-conditional perturbation (proposed-perturb (class-conditional)) performs best among all the comparison targets. Especially in MNIST, the error rate of 'proposed-perturb (class-conditional)' with 2k per-class training examples is less than the error rate of all

Table 1: Error rate  $(\%)$  on the test set using the model with the best performance on the validation set. Numbers on the first row of each sub-table are the number of randomly chosen per-class training examples. The average performance of three different random-split datasets is described in this table (error rate on each random set is summarized in Appendix). Performance of three previous approaches (with gray background) and the proposed methods (baseline, random perturbation, and class-conditional perturbation in order) is summarized.  

<table><tr><td>MNIST (# train examples per class)</td><td>10</td><td>20</td><td>50</td><td>100</td><td>200</td><td>500</td><td>1k</td><td>2k</td><td>(all) 5k</td></tr><tr><td>feed-forward model; Figure 2(a)</td><td>24.55</td><td>16.00</td><td>10.35</td><td>6.58</td><td>4.71</td><td>2.94</td><td>1.90</td><td>1.45</td><td>1.04</td></tr><tr><td>joint learning model with recon-one; Figure 2(b)</td><td>21.67</td><td>13.60</td><td>7.85</td><td>5.44</td><td>4.14</td><td>2.50</td><td>1.84</td><td>1.45</td><td>1.12</td></tr><tr><td>joint learning model with recon-all; Figure 2(b)</td><td>20.11</td><td>13.69</td><td>9.15</td><td>6.77</td><td>5.39</td><td>3.89</td><td>2.91</td><td>2.28</td><td>1.87</td></tr><tr><td>proposed-base; Figure 1(b)</td><td>21.35</td><td>11.65</td><td>6.33</td><td>4.32</td><td>3.07</td><td>1.98</td><td>1.29</td><td>0.94</td><td>0.80</td></tr><tr><td>proposed-perturb (random); Figure 1(c)</td><td>20.17</td><td>11.68</td><td>6.24</td><td>4.12</td><td>3.04</td><td>1.88</td><td>1.24</td><td>0.96</td><td>0.65</td></tr><tr><td>proposed-perturb (class-conditional); Figure 1(c)</td><td>20.11</td><td>10.59</td><td>5.92</td><td>3.79</td><td>2.72</td><td>1.78</td><td>1.15</td><td>0.88</td><td>0.62</td></tr><tr><td>CIFAR-10 (# train examples per class)</td><td>10</td><td>20</td><td>50</td><td>100</td><td>200</td><td>500</td><td>1k</td><td>2k</td><td>(all) 4k</td></tr><tr><td>feed-forward model; Figure 2(a)</td><td>73.82</td><td>68.99</td><td>61.30</td><td>54.93</td><td>46.97</td><td>33.69</td><td>26.63</td><td>20.97</td><td>17.80</td></tr><tr><td>joint learning model with recon-one; Figure 2(b)</td><td>75.68</td><td>69.05</td><td>61.44</td><td>55.02</td><td>46.18</td><td>33.62</td><td>26.78</td><td>21.25</td><td>17.68</td></tr><tr><td>joint learning model with recon-all; Figure 2(b)</td><td>73.33</td><td>67.63</td><td>62.59</td><td>56.37</td><td>50.51</td><td>41.26</td><td>32.55</td><td>26.38</td><td>22.71</td></tr><tr><td>proposed-base; Figure 1(b)</td><td>71.63</td><td>66.17</td><td>58.91</td><td>52.65</td><td>43.46</td><td>31.86</td><td>25.76</td><td>21.06</td><td>17.45</td></tr><tr><td>proposed-perturb (random); Figure 1(c)</td><td>71.69</td><td>66.75</td><td>58.95</td><td>53.01</td><td>43.71</td><td>31.80</td><td>25.50</td><td>20.81</td><td>17.43</td></tr><tr><td>proposed-perturb (class-conditional); Figure 1(c)</td><td>71.50</td><td>66.87</td><td>58.30</td><td>52.32</td><td>42.98</td><td>30.91</td><td>24.81</td><td>20.19</td><td>16.16</td></tr></table>

types of previous works with the entire training set (approximately 5k per-class examples). More empirical analyses in terms of the generalization performance are handled in next subsection.

# 4.4 QUALITATIVE ANALYSIS

As mentioned before, random perturbation by adding unstructured noise directly to the latent representation easily destroys the semantic feature of the original representation. We compared two different perturbation methods (random and class-conditional) by visualizing the examples reconstructed from the perturbed latent vectors (Figure 5). Top row is the original examples selected from training set (among  $2\mathrm{k}$  per-class training examples), and the rest are the reconstructions of their perturbed latent representations. Based on the architecture described in Figure 1(b), we generated five different perturbed latent representations according to the type of perturbation, and reconstructed the perturbed latent vectors through decoding path for reconstruction.

Figure 5(a) and (b) show the examples reconstructed from the random and class-conditional perturbations, respectively. For both cases, zero-mean Gaussian random noise (0.2 standard deviation) is used for perturbation. As shown in Figure 5(a), random perturbation cannot guarantee the preservation of original semantics; for example, semantics of '1' is mostly destroyed under random perturbation, and some examples of '3' are reconstructed as being similar to '8' rather than its original content '3'. Figure 5(b) shows the examples reconstructed from the class-conditionally perturbed representation. The reconstructed examples show subtle semantic variations while maintaining the original semantic contents; for example, thickness difference in '3' (example on the third row) or writing style difference in '8' (openness of the top left corner).

Figure 6 shows the overall effect of the perturbation. In this analysis, 100 per-class MNIST examples are used for training. From the trained model based on the architecture described in Figure 1(b), latent representations  $z$  of all the 50k examples (among 50k examples, only 1k examples were used for training) are visualized by using t-SNE (Maaten & Hinton, 2008). Only the training examples of three classes (0, 1, and 9) among ten classes are depicted as black circles for visual discrimination in Figure 6(a). The rest of the examples which were not used for training (approximately 4.9k examples per class) are depicted as a background with different colors. We treat the colored

![](images/98f910375eff54314fdfa6fbd770939da09dd21a5826d82b5d1ac2443530e32b.jpg)  
(a)

![](images/29f8278b561fc5dd0aabfbe77aa190fcea71fb756a77054132247ef7c8ec5074.jpg)  
(b)

![](images/fa75b0b1706b30ab23726ff4cc53babf16ad9c5933d6e913271f0ce7501f493e.jpg)  
(a)

![](images/d21b19d93346b2ec0d1dd564e4eadade913f2b6c8277494f0585c35e827cde28.jpg)  
Figure 5: Examples reconstructed from (a) randomly, and (b) class-conditionally perturbed latent vectors (top row shows the original training examples).  
(b)

![](images/ce5c2b6f51ed8ad07a458f77401439f0358cafb6756ccb687bf9d4726d1532a8.jpg)  
(c)  
Figure 6: Training examples (circles or crosses with colors described below) over the examples not used for training (depicted as background with different colors); (a) training examples (black circles), (b) training examples (yellow circles) with  $3 \times$  randomly perturbed samples (blue crosses), and (c) training examples (yellow circles) with  $3 \times$  class-conditionally perturbed samples (blue crosses). Best viewed in color.

background examples (not used for training) as a true distribution of unseen data in order to estimate the generalization level of learned representation according to the type of perturbation. Figure 6(b) and (c) show the training examples (100 examples per class with yellow circles) and their perturbed ones ( $3 \times$  sampled from each example with blue crosses) through random and class-conditional perturbations, respectively.

In Figure 6(b), perturbed samples are distributed near the original training examples, but some samples outside the true distribution cannot be identified easily with appropriate classes. This can be explained with Figure 5(a), since some perturbed samples are ambiguous semantically. In Figure 6(c), however, most of the perturbed samples evenly cover the true distribution. As mentioned before, stochastic perturbation with the class-conditional additive noise during training implicitly incurs the effect of augmentation on the latent space while resulting in better generalization. Per-class t-SNE results are summarized in Appendix (A3.1).

# 5 DISCUSSION

We introduced a novel latent space modeling method for supervised tasks based on the standard feed-forward neural network architecture. The presented model simultaneously optimizes both supervised and unsupervised losses based on the assumption that the better latent representation can be obtained by maximizing the total correlation of all the random variables defined by the standard feed-forward neural network model. Especially the stochastic perturbation process which is achieved by modeling the class-conditional additive noise during training enhances the represent

tational power of the latent space. From the proposed semantic noise modeling process, we can expect improvement of generalization performance in supervised learning with implicit semantic augmentation effect on the latent space.

The presented model architecture can be intuitively extended to semi-supervised learning because it is implemented as the joint optimization of supervised and unsupervised objectives. For semi-supervised learning, however, logical link between features learned from labelled and unlabelled data needs to be considered additionally. We leave the extension of the presented approach to semi-supervised learning for the future.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL http://tensorflow.org/. Software available from tensorflow.org.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In International Conference on Learning Representations (ICLR), 2015.  
Yoshua Bengio, Pascal Lamblin, Dan Popovici, Hugo Larochelle, et al. Greedy layer-wise training of deep networks. In Advances in Neural Information Processing Systems (NIPS), 2007.  
Kyunghyun Cho and Xi Chen. Classifying and visualizing motion capture sequences using deep neural networks. In International Conference on Computer Vision Theory and Applications, 2014.  
Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In International Conference on Machine Learning (ICML), 2008.  
Ian Goodfellow, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. Multi-prediction deep boltzmann machines. In Advances in Neural Information Processing Systems (NIPS), 2013.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In International conference on acoustics, speech and signal processing, 2013.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Computer Vision and Pattern Recognition (CVPR), 2016.  
Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. Signal Processing Magazine, IEEE, 29(6):82-97, 2012.  
Geoffrey E. Hinton, Simon Osindero, and Yee Whye Teh. A fast learning algorithm for deep belief nets. Neural Computation, 18:1527-1554, 2006.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems (NIPS), 2012.  
Hugo Larochelle and Yoshua Bengio. Classification using discriminative restricted boltzmann machines. In International Conference on Machine Learning (ICML), 2008.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research (JMLR), 9(Nov):2579-2605, 2008.  
Jonathan Masci, Ueli Meier, Dan Ciresan, and Jürgen Schmidhuber. Stacked convolutional autoencoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks, 2011.  
Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, and Tapani Raiko. Semi-supervised learning with ladder networks. In Advances in Neural Information Processing Systems (NIPS), 2015.  
Ruslan Salakhutdinov and Geoffrey E Hinton. Deep boltzmann machines. In Artificial Intelligence and Statistics Conference (AISTATS), 2009.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations (ICLR), 2015.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning Research (JMLR), 11:3371-3408, 2010.  
Satosi Watanabe. Information theoretical analysis of multivariate correlation. IBM Journal of research and development, 4(1):66-82, 1960.  
Yuting Zhang, Kibok Lee, and Honglak Lee. Augmenting supervised neural networks with unsupervised objectives for large-scale image classification. In International Conference on Machine Learning (ICML), 2016.  
Junbo Zhao, Michael Mathieu, Ross Goroshin, and Yann Lecun. Stacked what-where auto-encoders. In International Conference on Learning Representations (ICLR), 2015.
