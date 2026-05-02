# ALTERNATING DIRECTION METHOD OF MULTIPLIERS FOR SPARSE CONVOLUTIONAL NEURAL NETWORKS

Farkhondeh Kiaee, Christian Gagne, and Mahdieh Abbasi

Computer Vision and Systems Laboratory

Department of Electrical Engineering and Computer Engineering

Université Laval, Québec, QC G1V 0A6, Canada

{farkhondeh.kiaee.1,mahdieh.abbasi.1}@ulaval.ca

christian.gagne@gel.ulaval.ca

# ABSTRACT

The storage and computation requirements of Convolutional Neural Networks (CNNs) can be prohibitive for exploiting these models over low-power or embedded devices. This paper reduces the computational complexity of the CNNs by minimizing an objective function, including the recognition loss that is augmented with a sparsity-promoting penalty term. The sparsity structure of the network is identified using the Alternating Direction Method of Multipliers (ADMM), which is widely used in large optimization problems. This method alternates between promoting the sparsity of the network and optimizing the recognition performance, which allows us to exploit the two-part structure of the corresponding objective functions. In particular, we take advantage of the separability of the sparsity-inducing penalty functions to decompose the minimization problem into sub-problems that can be solved sequentially. Applying our method to a variety of state-of-the-art CNN models, our proposed method is able to simplify the original model, generating models with less computation and fewer parameters, while maintaining and often improving generalization performance. Accomplishments on a variety of models strongly verify that our proposed ADMM-based method can be a very useful tool for simplifying and improving deep CNNs.

# 1 INTRODUCTION

Deep Convolutional Neural Networks (CNNs) have achieved remarkable performance in challenging computer vision problems such as image classification and object detection tasks, at the cost of a large number of parameters and computational complexity. These costs can be problematic for deployment especially on mobile devices and when real time operation is needed.

To improve the efficiency of CNNs, several attempts have been made to reduce the redundancy in the network. Gupta et al. (2015) and Vanhoucke et al. (2011) showed that further quantization of network weights that were originally represented as 32-bit floating point numbers can result in significant speedup with minimal loss of accuracy. Jaderberg et al. (2014) proposed to represent the full-rank original convolutional filters tensor by a low-rank approximation composed of a sequence of two regular convolutional layers, with rectangular filters in the spatial domain. They achieve a  $4.5\mathrm{x}$  speedup using different tensor decomposition schemes, with less than  $1\%$  dip in accuracy in a text recognition application. A different network connection structure is suggested by Ioannou et al. (2015), which implicitly learns linear combinations of rectangular filters in the spatial domain, with different vertical/horizontal orientations. Tai et al. (2015) presented an exact and closed form solution to the low-rank decomposition approach of Jaderberg et al. (2014) to enforce connection sparsity on CNNs. In most of these studies, the key observation is that the introduction of sparsity leads to a slight drop in performance but reduces significantly the network computation requirements.

Sparse learning has been shown to be efficient at pruning the irrelevant parameters in many practical applications, by incorporating sparsity-promoting penalty functions into the original problem, where the added sparsity-promoting terms penalize the number of parameters (Kiaee et al. (2016a;b;c)). Motivated by learning efficient architectures of a deep CNN for embedded implementations, our

work focuses on the design of a sparse network using an initial pre-trained dense CNN. Our approach aims at finding a sparsity structure that strikes a balance between performance and computation requirements of the deep networks.

The alternating direction method of multipliers (ADMM) (Boyd et al. (2011)) has been extensively studied to minimize the augmented Lagrangian function for optimization problems, by breaking them into smaller pieces. It turns out that ADMM has been recently applied in a variety of contexts (Lin et al. (2013a); Shen et al. (2012); Meshi & Globerson (2011)). We demonstrate that the ADMM provides an effective tool for optimal sparsity imposing on deep neural connections. This is achieved by augmenting a sparsity-inducing penalty term to the recognition loss of a pre-trained network. Different functions including the  $l_{0}$ -norm and its convex  $l_{1}$ -norm relaxations can be considered as a penalty term. The variables are then partitioned into two subsets, playing two different roles:

1. Promoting the sparsity of the network at the level of a predetermined sparse block structure;  
2. Optimizing the recognition error.

The augmented Lagrangian function is then minimized with respect to each subset by fixing all other subsets at each iteration. In the absence of the penalty term, the performance results correspond to the original network with a dense structure. By gradually increasing the regularization factor of the sparsity-promoting penalty term, the optimal parameters move from their initial setting to the sparse structure of interest. This regularization factor is increased until the desired balance between performance and sparsity is achieved.

Our numerical experiments on three benchmark datasets, namely CIFAR-10, CIFAR-100, and SVHN, show that the structure of the baseline networks can be significantly sparsified. While most previous efforts report a small drop or no change in performance, we found a slight increase of classification accuracy in some cases.

This paper is organized as follows. In Section 2, the design problem of sparse convolutional neural network is formulated. In Section 3, we present the ADMM algorithm as a flexible approach to decompose the objective function and provide the solution directions to the sparsity-promoting and performance promoting sub-problems. Several experiments are provided in Section 4 to demonstrate the effectiveness of the developed approach. This paper concludes in Section 5.

# 2 CNN WITH SPARSE FILTERS

Consider a CNN network consisting of a total of  $L$  layers, including convolutional and fully connected layers, which are typically interlaced with rectified linear units and pooling (Fig. 1). Let the  $l$ -th layer includes  $m^l$  input feature maps and  $n^l$  output feature maps, with  $W_{ij}^{l}$  representing the convolution filter between the  $i$ -th and  $j$ -th input and output feature maps, respectively<sup>1</sup>. Our goal is to design the optimal filters, subject to sparse structural constraints. In order to obtain the filters which balance a trade-off between the minimization of the loss function and sparseness, we consider the following objective function

$$
\underset {\boldsymbol {W}} {\text {m i n i m i z e}} \mathcal {L} _ {\text {n e t}} (\boldsymbol {W}) + \mu f (\boldsymbol {W}), \tag {1}
$$

where  $\mathcal{L}_{net}$  stands for the logistic loss function of the output layer of the network which is a function of the convolutional filters of all layers  $\pmb{W} = \{\pmb{W}_{ij}^{l}|i = 1,\dots,m^{l},j = 1,\dots,n^{l},l = 1,\dots,L\}$ . The term  $f(\pmb{W})$  is a penalty function on the total size of the filters. The  $l_{0}$ -norm (cardinality) function or relaxations to higher orders such as  $l_{1}$ -norm function can be employed to promote the sparsity of the filters.

The parameter  $\mu$  controls the effect of sparse penalty term. As  $\mu$  varies, the solution of (1) traces the trade-off path between the performance and the sparsity. In the next section, the alternating direction method of multipliers (ADMM) which is employed to find the optimal solution of (1) is described.

![](images/cc98e262fc612fad2d924944d7351eecad768aa4acd0e090a209bee1e415d69d.jpg)  
Figure 1: Architecture of a typical CNN, selected sparsity blocks at convolutional and fully connected layers are shown in blue.

# 3 USING ADMM FOR OPTIMAL SPARSIFYING OF THE CNNS

Consider the following constrained optimization problem:

$$
\underset {\boldsymbol {W}, \boldsymbol {F}} {\text {m i n i m i z e}} \quad \mathcal {L} _ {n e t} (\boldsymbol {W}) + \mu f (\boldsymbol {F}),
$$

$$
\text {s . t .} \quad \boldsymbol {W} - \boldsymbol {F} = \mathbf {0}, \tag {2}
$$

which is clearly equivalent to the problem stated in (1). The key point here is that by introducing an additional variable  $\pmb{F}$  and an additional constraint  $\pmb{W} - \pmb{F} = \mathbf{0}$ , the objective function of the problem (1) is decoupled into two parts that depend on two different variables.

The augmented Lagrangian associated with the constrained problem (2) is given by

$$
\begin{array}{l} \mathcal {C} (\boldsymbol {W}, \boldsymbol {F}, \Gamma) = \mathcal {L} _ {\text {n e t}} (\boldsymbol {W}) + \mu f (\boldsymbol {F}) \\ + \sum_ {l, i, j} \operatorname {t r a c e} \left(\boldsymbol {\Gamma} _ {i j} ^ {l} ^ {T} \left(\boldsymbol {W} _ {i j} ^ {l} - \boldsymbol {F} _ {i j} ^ {l}\right)\right) + \frac {\rho}{2} \sum_ {l, i, j} \| \boldsymbol {W} _ {i j} ^ {l} - \boldsymbol {F} _ {i j} ^ {l} \| _ {F}, \tag {3} \\ \end{array}
$$

where  $\Gamma_{ij}^{l}$  is the dual variable (i.e., the Lagrange multiplier),  $\rho$  is a positive scalar,  $\| .\| _F$  and is the Frobenius norm.

In order to find a minimizer of the constrained problem (3), the ADMM algorithm uses a sequence of iterative computations:

1. Make use of a descent method to solve the following performance promoting problem,

$$
\boldsymbol {W} ^ {\{k + 1 \}} = \underset {\boldsymbol {W}} {\arg \min } \mathcal {C} \left(\boldsymbol {W}, \boldsymbol {F} ^ {\{k \}}, \boldsymbol {\Gamma} ^ {\{k \}}\right); \tag {4}
$$

2. Find the analytical expressions for the solutions of the following sparsity promoting problem.

$$
\boldsymbol {F} ^ {\{k + 1 \}} = \underset {\boldsymbol {F}} {\arg \min } \mathcal {C} \left(\boldsymbol {W} ^ {\{k + 1 \}}, \boldsymbol {F}, \boldsymbol {\Gamma} ^ {\{k \}}\right); \tag {5}
$$

3. Update the dual variable  $\Gamma_{ij}^{l}$  using a step-size equal to  $\rho$ , in order to guarantee that the dual feasibility conditions is satisfied in each ADMM iteration,

$$
\boldsymbol {\Gamma} _ {i j} ^ {l} ^ {\{k + 1 \}} = \boldsymbol {\Gamma} _ {i j} ^ {l} ^ {\{k \}} + \rho \left(\boldsymbol {W} _ {i j} ^ {l} ^ {\{k + 1 \}} - \boldsymbol {F} _ {i j} ^ {l} ^ {\{k + 1 \}}\right). \tag {6}
$$

Algorithm 1 Outline of the proposed sparse CNN algorithm  
1: function SPARSE-CNN(data, model)  
2: Set  $W$  to a pre-trained reference CNN model  
3:  $\Gamma = 0$ ,  $F = W$   
4:  $S$ : a set of small logarithmically spaced points in increasing order, as regularization factor.  
5: for each  $\mu$  in  $S$  do  
6: do  
7: Find the estimate of  $W^{\{k+1\}}$  by minimizing (7)  
8: Find the estimate of  $F^{\{k+1\}}$  from (9) or (10)  
9: Update dual variable  $\Gamma^{\{k+1\}}$  from (6)  
10: while  $\| W^{\{k+1\}} - F^{\{k+1\}} \|_F > \epsilon$  or  $\| F^{\{k+1\}} - F^{\{k\}} \|_F > \epsilon$   
11: Fix the identified sparse structure and fine-tune network according to  $\mathcal{L}_{net}$  w.r.t. non-zero parameters  
12: end for  
13: return  $W_{ij}^{l}$   
14: end function

The three described computation steps are applied in an alternating manner. Re-estimation stops when the Frobenius distance of  $\pmb{F}$  in two consecutive iterations as well as the Frobenius distance of  $\pmb{W}$  and  $\pmb{F}$  at current iterations are less than a small threshold value. The details of steps 1 and 2 are described in the next sections. The outline of the proposed sparse CNN approach is summarized in Algorithm 1. At each individual regularization  $\mu$ , in order to improve the performance of the sparse-structured network we fine-tune the initial non-augmented recognition loss subject to the parameters belonging to the identified sparse structure.

# 3.1 PERFORMANCE PROMOTING STEP

By completing the squares with respect to  $\mathbf{W}$  in the augmented Lagrangian  $\mathcal{C}(\mathbf{W},\mathbf{F},\Gamma)$ , we obtain the following equivalent problem to (4)

$$
\underset {\boldsymbol {W}} {\operatorname {m i n i m i z e}} \mathcal {L} _ {\text {n e t}} (\boldsymbol {W}) + \frac {\rho}{2} \sum_ {l, i, j} \| \boldsymbol {W} _ {i j} ^ {l} - \boldsymbol {U} _ {i j} ^ {l} \| _ {F} ^ {2}, \tag {7}
$$

where  $U_{ij}^{l} = F_{ij}^{l} - \frac{1}{\rho}\Gamma_{ij}^{l}$ . From (7), it can be seen that by exploiting the separability property of ADMM method in the minimization of the augmented Lagrangian, the sparsity penalty term is excluded from (7). The recognition loss function  $\mathcal{L}_{net}(W)$  is typically differentiable with respect to the parameters; this is in contrast to some choices of sparsity penalty terms (e.g.,  $l_0$ -norm which is a non-differentiable function). Consequently, descent algorithms that rely on the differentiability can be utilized to solve the performance promoting sub-problem (7) while different functions (e.g.,  $l_0$ -norm and  $l_1$ -norm) can be incorporated as means of sparsity penalty terms in the original problem (1).

This property allows that popular software and toolkit resources for Deep Learning, including Caffe, Theano, Torch, and TensorFlow, to be employed for implementing the proposed approach. In our work, we use Stochastic Gradient Descent (SGD) method of TensorFlow to optimize the weights  $(W)$ , which seemed a reasonable choice for the high-dimensional optimization problem at hand. The entire procedure relies mainly on the standard forward-backward pass that is used to train the convolutional network.

# 3.2 SPARSITY PROMOTING STEP

The completion of squares with respect to  $F$  in the augmented Lagrangian can be used to show that (5) is equivalent to

$$
\underset {\boldsymbol {F}} {\operatorname {m i n i m i z e}} \mu f (\boldsymbol {F}) + \frac {\rho}{2} \sum_ {l, i, j} \| \boldsymbol {F} _ {i j} ^ {l} - \boldsymbol {V} _ {i j} ^ {l} \| _ {F} ^ {2}, \tag {8}
$$

where  $\pmb{V}_{ij}^{l} = \pmb{W}_{ij}^{l} + \frac{1}{\rho}\pmb{\Gamma}_{ij}^{l}$ . From (8), it can be seen that the proposed method provides a flexible framework to select arbitrary sparsity blocks. Sparse structure can then be achieved at the level of

the selected block. Specifically, both terms on the right hand side of (8),  $f(\mathbf{F})$  (for either the case of  $l_{1}$ -norm or  $l_{0}$ -norm) as well as the square of the Frobenius norm can be written as a summation of component-wise functions of a tensor. In our experiments, individual filter components are selected as the sparsity blocks (see Fig. 1). Hence, (8) can simply be expressed in terms of  $F_{ij}^{l}$  components corresponding to the filters. However, any other individual sub-tensor components can be selected as the sparsity block.

More precisely, if  $f(\mathbf{F})$  is selected to be the  $l_{1}$ -norm function, then  $\mathcal{C}(\mathbf{F}) = \sum_{l,i,j} (\mu \| \mathbf{F}_{ij}^{l} \|_{F} + \frac{\rho}{2} \| \mathbf{F}_{ij}^{l} - \mathbf{V}_{ij}^{l} \|_{F}^{2})$  and consequently, (8) is converted to a minimization problem that only involves spatial filters. The solution of (8) can then be determined analytically by the following soft thresholding operation,

$$
\boldsymbol {F} _ {i j} ^ {l} * = \left\{ \begin{array}{l l} \left(1 - \frac {a}{\| \boldsymbol {V} _ {i j} ^ {l} \| _ {F}}\right) \boldsymbol {V} _ {i j} ^ {l}, & \text {i f} \| \boldsymbol {V} _ {i j} ^ {l} \| _ {F} > a \\ 0, & \text {o t h e r w i s e} \end{array} , \right. \tag {9}
$$

where  $a = \frac{\mu}{\rho}$ . Similarly, the following hard thresholding operation is the analytical solution for the case of the selection of the  $l_0$ -norm  $f(\pmb{F})$  penalty term.

$$
\boldsymbol {F} _ {i j} ^ {l} ^ {*} = \left\{ \begin{array}{l l} \boldsymbol {V} _ {i j} ^ {l}, & \text {i f} \| \boldsymbol {V} _ {i j} ^ {l} \| _ {F} > b \\ 0, & \text {o t h e r w i s e} \end{array} , \right. \tag {10}
$$

where  $b = \sqrt{\frac{2\mu}{\rho}}$

# 3.3 CONVERGENCE OF THE PROPOSED ADMM-BASED SPARSE CNN METHOD

For convex problems, the ADMM is guaranteed to converge to the global optimum solution (Boyd et al. (2011)). For non-convex problems, where there is a general lack of theoretical proof, extensive computational experience suggests that ADMM works well when the penalty parameter  $\rho$  in the augmented Lagrangian is chosen to be sufficiently large. This is related to the quadratic term that tends to locally convexify the objective function for sufficiently large  $\rho$ .

Unfortunately, in the deep learning problems, the objective is inherently highly non-convex and consequently there is the risk that it becomes trapped into a local optimum. This difficulty could be circumvented by considering a warm start that may be obtained by running a pre-trained version of the network. The proposed ADMM approach is then used to sparsify the final solution. Using this procedure, as the experiments in the next section show, we have obtained good empirical results.

# 4 EXPERIMENTAL RESULTS

In order to validate our approach, we show that our proposed sparse CNN approach can be efficiently applied to existing state-of-the-art network architectures to reduce the computational complexity without reducing the accuracy performance. For this purpose, we evaluate the proposed scheme on the CIFAR-10, CIFAR-100, and SVHN datasets with several CNN models.

# 4.1 RESULTS ON CIFAR-10 OBJECT CLASSIFICATION

The CIFAR-10 dataset is a well-known small dataset of 60,000 32 x 32 images in 10 classes. This dataset comprises standard sets of 50,000 training images, and 10,000 test images. As a baseline for the CIFAR-10 dataset, we deploy four models: the Network in Network (NIN) architecture (Lin et al., 2013b), its low-rank version (Ioannou et al., 2015), a custom CNN, and its low-rank counterpart as well, two last being learned from scratch on the CIFAR dataset. The configurations of the baseline models are outlined in Table 1.

The architecture of the NIN model is slightly different from the one introduced in Lin et al. (2013b). The original NIN uses 5x5 filters in the first and second convolutional layer which are replaced with one and two layers of 3x3 filters, respectively. As suggested by Ioannou et al. (2015), this modified architecture has comparable accuracy and less computational complexity. In the low-rank networks, every single convolutional layer of the full-rank model is replaced with two convolutional

Table 1: Structure of the baseline Networks.  

<table><tr><td></td><td>NIN</td><td>Low-rank NIN</td></tr><tr><td>conv1</td><td>3 × 3 × 192</td><td>h: 1 × 3 × 96
v: 3 × 1 × 96</td></tr><tr><td>conv2,3</td><td colspan="2">1 × 1 × 160, 1 × 1 × 96</td></tr><tr><td>conv4</td><td>3 × 3 × 192</td><td>h: 1 × 3 × 96
v: 3 × 1 × 96</td></tr><tr><td>conv5</td><td>3 × 3 × 192</td><td>h: 1 × 3 × 96
v: 3 × 1 × 96</td></tr><tr><td>conv6,7</td><td colspan="2">1 × 1 × 192, 1 × 1 × 192</td></tr><tr><td>conv8</td><td>3 × 3 × 192</td><td>h: 1 × 3 × 96
v: 3 × 1 × 96</td></tr><tr><td>conv9,10</td><td colspan="2">1 × 1 × 192, 1 × 1 × 10</td></tr></table>

<table><tr><td></td><td>CNN</td><td>Low-rank CNN</td></tr><tr><td>conv1</td><td>3 × 3 × 96</td><td>h: 1 × 3 × 48
v: 3 × 1 × 46</td></tr><tr><td>conv2</td><td>3 × 3 × 128</td><td>h: 1 × 3 × 64
v: 3 × 1 × 64</td></tr><tr><td>conv3</td><td>3 × 3 × 256</td><td>h: 1 × 3 × 128
v: 3 × 1 × 128</td></tr><tr><td>conv4</td><td>3 × 3 × 64</td><td>h: 1 × 3 × 32
v: 3 × 1 × 32</td></tr><tr><td>fc1</td><td>1024 × 256</td><td>1024 × 256</td></tr><tr><td>fc2</td><td>256 × 10</td><td>256 × 10</td></tr></table>

layers with horizontal and vertical filters. NIN and low-rank NIN have an accuracy of  $90.71\%$  and  $90.07\%$ , respectively. The custom CNN and its low-rank variant show a baseline accuracy of  $80.0\%$  and  $80.2\%$ , respectively.

The results of our experiments are plotted in Fig. 2 for both  $l_{0}$ -norm and  $l_{1}$ -norm sparsity constraints. In the implementation of the performance promoting step in Section 3.1, the batch size is 128 and the learning rate is set to a rather small value (i.e., 0.001 to search the space around the dense initialized filters to find a sparse solution).

Fig. 2 shows how the accuracy performance changes as we increase the regularization factor  $\mu$ . The case with  $\mu = 0$  can be considered as the baseline model. In order to avoid over pruning of some layers, if the number of pruned filters in one layer exceeds  $50\%$  of the total number of filters in that layer, then we change the pruning threshold to the statistical mean of the Frobenius norm of all the filters at that layer in the sparsity promoting step (Section 3.2) to stop the over pruning of that layer. Taking the NIN and low-rank-NIN as an example, using the  $l_{0}$ -norm sparsity function, the parameters in the networks are reduced by  $34.13\%$  and  $28.5\%$  and the relative accuracy performance is  $+5\%$  and  $+1.23\%$ , respectively. Using the  $l_{1}$ -norm sparsity constraint achieves slightly lower accuracy compared to the  $l_{0}$ -norm, although it still conveniently sparsifies the network.

Using the proposed sparsity promoting approach on the custom CNN models, the networks with sparse connections and similar accuracy (79.9% v.s. 80%) are achieved, but they have approximately 49.4% fewer parameters than the original networks model. For the low-rank CNN, we achieve a comparable accuracy of 80.14%, with 25% fewer parameters.

In Appendix A, we show the details of changing sparsity in different layers of the networks. According to the results presenting in Table 2 of Appendix A, the number of parameters in the network can be reduced by a large factor, especially for the higher convolution layers. Interestingly, even with significant reductions in the number of parameters, the performance does not decrease that much. Note that most of the results listed in Table 2 outperform the baseline model.

# 4.2 RESULTS ON CIFAR-100 OBJECT CLASSIFICATION

The CIFAR-100 dataset is similar to the CIFAR-10 dataset containing 100 classes with 600 images per class. For CIFAR-100 we again use the baseline networks in Table 1 with only one structural difference (i.e., the NIN networks contain 100 feature maps at the last convolution layer and custom CNN networks contain 100 output labels). The baseline NIN, low-rank NIN, custom CNN, and low-rank CNN models show a test accuracy of  $63.3\%$ ,  $63.6\%$ ,  $60.11\%$ , and  $60.23\%$ , respectively. Using the proposed sparsity promoting approach on these networks, the total number of parameters in the layers can be reduced by a large factor with comparable or even better performance accuracy.

In particular, on the CIFAR-100 dataset, we achieve  $64.09\%$  classification accuracy with  $34.1\%$  sparsity for the NIN model, which improves upon the original NIN on this dataset. A test accuracy of  $65.23\%$  is obtained for CIFAR-100 for the low-rank NIN model with  $28.5\%$  sparsity which surpasses the performance of the baseline model. The proposed method on custom CNN and low-rank CNN show comparable performance accuracy to their corresponding baseline models (59.82% vs 60.11%

![](images/d1e514a21afb927adfe5b3c01155a2d02ea06361e17f79f5ad1580a622a2fc4c.jpg)  
Figure 2: Variation of the accuracy measure (odd rows) against the values of  $\mu_0$  parameter and (even rows) against the normalized number of zero elements for different models and datasets.

and  $60.1\%$  vs  $60.23\%$  ) with much less computation (49.7% and  $24.4\%$  number of zero elements, respectively). The details of changing sparsity in different layers of the networks on the CIFAR-100 dataset are presented in Table 3 of Appendix A. The same conclusions made for CIFAR-10 can be drawn from these results.

# 4.3 RESULTS ON SVHN OBJECT CLASSIFICATION

The SVHN dataset consists of 630,420 32x32 color images of house numbers collected by Google Street View. The task of this dataset is to classify the digit located at the center of each image. The structure of the baseline models used in SVHN is similar to those used for CIFAR-10, which are presented in Table 1. The training and testing procedure of the baseline models follows Lin et al. (2013b). The baseline NIN, low-rank NIN, custom CNN and low-rank CNN models show the accuracy of  $96.2\%$ ,  $96.7\%$ ,  $85.1\%$ , and  $87.6\%$ , respectively. For this dataset, by applying our proposed sparse approach to NIN and low-rank NIN models, we obtain a higher accuracy of  $96.97\%$  and  $98\%$  with  $34.17\%$  and  $28.6\%$  fewer parameters, respectively. We also achieve comparable accuracy of  $83.3\%$  and  $86.3\%$  using  $49.7\%$  and  $24.7\%$  less parameters of the original model parameters on custom CNN and low-rank CNN models, respectively (see Table 4 of Appendix A for the details on changing the sparsity in different layers of the networks on SVHN dataset).

# 5 DISCUSSION AND CONCLUSION

In this paper we proposed a framework to optimally sparsify a pre-trained CNN approach. We employed the ADMM algorithm to solve the optimal sparsity-promoting problem, whose solution gradually moves from the original dense network to the sparse structure of interest as our emphasis on the sparsity-promoting penalty term is increased.

The proposed method could potentially reduce the memory and computational complexity of the CNNs significantly. Briefly, the main contributions of the proposed sparse CNN can be summarized as follows:

1. It provides a flexible framework to select arbitrary sparsity blocks. Any individual sub-tensor element can be selected as the sparsity block.  
2. The sparsity-promoting penalty function (unlike the recognition loss) is separable with respect to the sparsity blocks. The solution to the sparsity-promoting sub-problem can then be determined analytically.  
3. The recognition loss function unlike the  $l_0$ -norm penalty term is differentiable with respect to the parameters; descent algorithms that rely on the differentiability can then be utilized to solve the sub-problem of recognition error minimization.  
4. Several approaches have developed to create sparse networks by applying pruning or sparsity regularizers (Liu et al. (2015); Han et al. (2015); Collins & Kohli (2014)). However, these approached require training the original full model and do not enjoy the separability properties and efficient training of the proposed ADMM-based approach.  
5. There are recent works focusing on reducing the parameters in the convolutional layers (Jaderberg et al. (2014); Ioannou et al. (2015); Tai et al. (2015)). In CNN models, the model size is dominated by the fully connected layers. The presented approaches are then not capable of reducing the size of the whole model. Our proposed approach can be applied on both the convolution and fully connected layers and can speed up the computation as well as compressing the size of the model.  
6. Several attempts have been made to compress the deep networks using the weights sharing and quantization (Han et al. (2016); Gupta et al. (2015); Vanhoucke et al. (2011)). However, these techniques can be used in conjunction with our proposed sparse method to achieve further speedup.

# ACKNOWLEDGMENTS

The authors gratefully acknowledge financial support by NSERC-Canada, MITACS and E Machine Learning Inc., a GPU grant from NVidia, and access to the computational resources of Calcul Quebec and Compute Canada. The authors are also grateful to Annette Schwerdtfeger for proofreading this manuscript.

# REFERENCES

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1-122, 2011.  
Maxwell D Collins and Pushmeet Kohli. Memory bounded deep convolutional networks. arXiv preprint arXiv:1412.1442, 2014.  
Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. In Proceedings of the 32nd International Conference on Machine Learning (ICML-15), pp. 1737-1746, 2015.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems, pp. 1135-1143, 2015.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding. International Conference on Learning Representations, 2016.  
Yani Ioannou, Duncan Robertson, Jamie Shotton, Roberto Cipolla, and Antonio Criminisi. Training CNNs with low-rank filters for efficient image classification. International Conference on Learning Representations, 2015.  
Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Speeding up convolutional neural networks with low rank expansions. In Proceedings of the British Machine Vision Conference (BMVC), 2014.  
Farkhondeh Kiaee, Christian Gagne, and Hamid Sheikhzadeh. A double-layer ELM with added feature selection ability using a sparse bayesian approach. Neurocomputing, 216:371 - 380, 2016a.  
Farkhondeh Kiaee, Hamid Sheikhzadeh, and Samaneh Eftekhari Mahabadi. Relevance vector machine for survival analysis. IEEE Trans. on Neural Networks and Learning Systems, 27(3):648-660, 2016b.  
Farkhondeh Kiaee, Hamid Sheikhzadeh, and Samaneh Eftekhari Mahabadi. Sparse bayesian mixed-effects extreme learning machine, an approach for unobserved clustered heterogeneity. Neurocomputing, 175:411-420, 2016c.  
Fu Lin, Makan Fardad, and Mihailo R Jovanovic. Design of optimal sparse feedback gains via the alternating direction method of multipliers. IEEE Transactions on Automatic Control, 58(9): 2426-2431, 2013a.  
Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. arXiv preprint arXiv:1312.4400, 2013b.  
Baoyuan Liu, Min Wang, Hassan Foroosh, Marshall Tappen, and Marianna Pensky. Sparse convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 806-814, 2015.  
Ofer Meshi and Amir Globerson. An alternating direction method for dual map LP relaxation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 470-483. Springer, 2011.  
Chao Shen, Tsung-Hui Chang, Kun-Yu Wang, Zhengding Qiu, and Chong-Yung Chi. Distributed robust multicell coordinated beamforming with imperfect CSI: An ADMM approach. IEEE Transactions on signal processing, 60(6):2988-3003, 2012.  
Cheng Tai, Tong Xiao, Xiaogang Wang, et al. Convolutional neural networks with low-rank regularization. arXiv preprint arXiv:1511.06067, 2015.  
Vincent Vanhoucke, Andrew Senior, and Mark Z Mao. Improving the speed of neural networks on CPUs. In Deep Learning and Unsupervised Feature Learning Workshop, NIPS2011, 2011.
