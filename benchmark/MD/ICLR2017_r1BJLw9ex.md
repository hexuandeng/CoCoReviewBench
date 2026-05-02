# ADJUSTING FOR DROPOUT VARIANCE IN BATCH NORMALIZATION AND WEIGHT INITIALIZATION

Dan Hendrycks*

University of Chicago dan@ttic.edu

Kevin Gimpel

Toyota Technological Institute at Chicago  
kgimpel@ttic.edu

# ABSTRACT

We show how to adjust for the variance introduced by dropout with corrections to weight initialization and Batch Normalization, yielding higher accuracy. Though dropout can preserve the expected input to a neuron between train and test, the variance of the input differs. We thus propose a new weight initialization by correcting for the influence of dropout rates and an arbitrary nonlinearity's influence on variance through simple corrective scalars. Since Batch Normalization trained with dropout estimates the variance of a layer's incoming distribution with some inputs dropped, the variance also differs between train and test. After training a network with Batch Normalization and dropout, we simply update Batch Normalization's variance moving averages with dropout off and obtain state of the art on CIFAR-10 and CIFAR-100 without data augmentation.

# 1 INTRODUCTION

Weight initialization and Batch Normalization greatly influence a neural network's ability to learn. Both methods can allow for a unit-variance neuron input distribution. This is desirable because variance larger or smaller than one may cause activation outputs to explode or vanish. In order to encourage unit-variance, early weight initialization attempts sought to adjust for a neuron's fan-in (LeCun et al., 1998). More recent initializations correct for a neuron's fan-out (Glorot & Bengio, 2010). Meanwhile, some weight initializations compensate for the compressiveness of the ReLU nonlinearity (the ReLU's tendency to reduce output variance) (He et al., 2015). Indeed, He et al. (2015) also show that initializations without a specific, small corrective factor can render a neural network untrainable. To address this issue Batch Normalization (Ioffe & Szegedy, 2015) reduces the role of weight initialization at the cost of up to  $30\%$  more computation (Mishkin & Matas, 2015). A less computationally expensive solution is the LSUV weight initialization, yet this still requires computing batch statistics, a special forward pass, and makes no adjustment for backpropagation error signal variance (Mishkin & Matas, 2015). Similarly, weight normalization uses a special feedforward pass and computes batch statistics (Salimans & Kingma, 2016). The continued development of variance stabilizing techniques testifies to its importance for neural networks.

Both Batch Normalization and previous weight initializations do not accommodate the variance introduced by dropout, and we contribute methods to fix this. First we demonstrate a new weight initialization technique which includes a new correction factor for a layer's dropout rate and adjusts for an arbitrary nonlinearity's effect on the neuron output variance. All of this is obtained without computing batch statistics or special adjustments to the forward pass, unlike recent methods to control variance (Ioffe & Szegedy, 2015; Mishkin & Matas, 2015; Salimans & Kingma, 2016). By this new initialization, we enable faster and more accurate convergence. Afterward, we show that networks trained with Batch Normalization can improve their accuracy by adjusting for dropout's variance. We accomplish this by training a network with both Batch Normalization and dropout, then after training we feed forward the training dataset with dropout off to reestimate the Batch Normalization variance estimates. Because of this simple, general technique, we obtain state of the art on CIFAR-10 and CIFAR-100 without data augmentation.

# 2 WEIGHT INITIALIZATION

# 2.1 DERIVATION

In this section, we derive our new initialization by considering a neuron input distribution and its major sources of variance. We accomplish this by separately considering the feedforward and the backpropagation stages.

# 2.1.1 THE FORWARD PASS

We use  $f$  to denote the pointwise nonlinearity in each neural network layer. For simplicity, we use the term "neuron" to refer to an entry in a layer before applying  $f$ . Let us also call the input of the  $l$ -th layer  $z^{l-1}$ , and let the  $n_{\mathrm{in}} \times n_{\mathrm{out}}$  weight matrix  $W^l$  map from layer  $l-1$  to  $l$ . Let an entry of this matrix be  $w^l$ . In our upcoming initialization, we initialize each column of  $W^l$  on the unit hypersphere so that each column has an  $\ell_2$  norm of 1. Now, if we assume that this network is trained with a dropout keep rate of  $p$ , we must scale the output of a layer by  $1/p$ . Also assume  $f(z^{l-1})$  and  $W$  are zero-centered. With that now specified, we conclude that neuron  $i$  of layer  $z^l$  has the variance

$$
\begin{array}{l} \operatorname {V a r} \left(z _ {i} ^ {l}\right) = \operatorname {V a r} \left(\sum_ {k = 1} ^ {n _ {\text {i n}}} W _ {k i} ^ {l} f \left(z ^ {l - 1}\right) _ {k} / p\right) \\ \approx \frac {n _ {\mathrm {i n}} p}{p ^ {2}} \operatorname {V a r} \left(w ^ {l} f \left(z ^ {l - 1}\right)\right) \\ = \mathbb {E} [ f (z ^ {l - 1}) ^ {2} ] / p \\ \end{array}
$$

because  $\mathrm{Var}(w^l) = 1 / n_{\mathrm{in}}$ , since we initialized  $W^{l}$ 's columns on the unit hypersphere. Knowing this variance allows us to adjust for the influence of an arbitrary nonlinearity and a desired dropout rate.

We empirically verify that a weight initialization with this forward correction allows for consistent input distribution variance throughout the layers of a 20-layer neural network for differing dropout rates. Specifically, we can encourage unit variance by dividing  $W$ , initialized on the unit hypersphere, by  $\sqrt{\mathbb{E}(f(z^{l-1})^2)/p}$ . Let us compare this correction to other initializations by feeding forward a random standard normal matrix through 20 layers. Figure 1 shows the results of such an experiment, and in the experiment we use a ReLU activation function. Of course, as He initialization was designed specifically for the ReLU, it performs well when  $p = 1$ , but has an exploding distribution when there is dropout. Only the initialization with a  $\sqrt{\mathbb{E}(f(z^{l-1})^2)/p}$  corrective term demonstrates stability when a feedforward does or does not use dropout.

# 2.1.2 BACK PROPAGATION

A similar analysis shows that if  $L$  is our loss function and  $\delta^l = \frac{\partial L}{\partial z^l}$ , then

$$
\operatorname {V a r} \left(\delta^ {l}\right) \approx p n _ {\text {o u t}} \operatorname {V a r} \left(w ^ {l + 1} \delta^ {l + 1} f ^ {\prime} \left(z ^ {l}\right)\right) = p \mathbb {E} \left[ f ^ {\prime} \left(z ^ {l}\right) ^ {2} \right].
$$

In appendix A we empirically verify that this backward correction allows for consistent backpropagation error signal variance throughout the layers of a 20-layer neural network for differing dropout rates.

# 2.2 OUR INITIALIZATION

We want that  $\operatorname{Var}(z^l) = 1$  and  $\operatorname{Var}(\delta^l) = \operatorname{Var}(\delta^{l + 1})$ . To meet these different goals, we can initialize our weights by adding these variances, while others take the arithmetic mean of these variances or ignore the backpropagation variance altogether (He et al., 2015; Glorot & Bengio, 2010). Therefore, if  $W^l$  has its columns sampled uniformly from the surface of a unit hypersphere or is an orthonormal matrix, then our initialization is

$$
W ^ {l} / \sqrt {\mathbb {E} [ f (z ^ {l - 1}) ^ {2} ] / p + p \mathbb {E} [ f ^ {\prime} (z ^ {l}) ^ {2} ]}.
$$

For convolutional neural networks, adjusting for the backpropagation signal is less common, so one could simply use the initialization  $W^{l} / \sqrt{\mathbb{E}[f(z^{l - 1})^{2}] / p}$  in this case. This initialization accounts for

![](images/6fc640639c6c79a79cfe5cd8472a8d178a5285d3bf7ed0f1f2bf83da8978461f.jpg)

![](images/01e7c65a57946be1205484c3575c3fffae2f51ae630b326dfd2276ab190ab689.jpg)

![](images/abdc27433aa6054f887e21d5d9344992aadf3b10f0d6369b4e308a625aa935d2.jpg)

![](images/6744f9b00047b321b3b5f5f00ff1d4621ecf08a8c2102c0f0d922ac89735216b.jpg)

![](images/90ff48ae898fe106b9eff9b0e1efb3c652ac8bf12457a8aa829026664c2735d0.jpg)

![](images/6506fea5bfff355ba2c09dfb2fb28fada78e551609f13495b44b8bc9fd1f9986.jpg)

![](images/74ec72812e54f3ecbe226b953b29f83474db8edee8f84cc06dc1357fec2dd518.jpg)

![](images/93296ab161f927e80b1493508b54b509490e8322fe3757369112c2dccb97831b.jpg)

![](images/9bd793494b245b487b75e8e62a778db679e9fee749d1c05b8c782e564eaab25f.jpg)

![](images/160ffc6c09f150d3adb81c5f1564e3a5ab105ba23a25d4bb8f33b9b54224e658.jpg)  
Figure 1: A comparison of a unit hypersphere initialization with a forward correction, the Xavier initialization, and the He initialization. Each plot shows the probability density function of a neuron's inputs and outputs across layers. In particular, the range of values vary widely between each initialization, with exponential blowups and decay for He and Xavier initializations, respectively. Values set to zero by dropout are removed from the probability density functions.

![](images/6b1f884c39f8867a1aa22a0bf4677ffbb42a0a565629a39d70b9e42e79bf5dbc.jpg)

![](images/77eecde6e4e04f2db17b72cd7996ab57b23ae16a02613a47239b473adbcfcc6e.jpg)

the influence of dropout rates and an arbitrary nonlinearity. Fortunately, we need only initialize a random standard Gaussian matrix and normalize its last dimension to generate  $W^l$ . Another strength of this initialization is that the expectations are similar to the values in Table 1 for standardized input data, so computing mini-batch statistics is needless for our initialization. We need only substitute in appropriate scalars during initialization. Let us now see these new adjustments in action.

<table><tr><td>Activation</td><td>E(f(zl-1)2)</td><td>E(f&#x27;(zl)2)</td></tr><tr><td>Identity</td><td>1</td><td>1</td></tr><tr><td>ReLU</td><td>0.5</td><td>0.5</td></tr><tr><td>SOI Map</td><td>0.5</td><td>0.5</td></tr><tr><td>GELU (μ=0, σ=1)</td><td>0.425</td><td>0.444</td></tr><tr><td>tanh</td><td>0.394</td><td>0.216</td></tr><tr><td>ELU (α=1)</td><td>0.645</td><td>0.671</td></tr></table>

Table 1: Activation adjustment estimates for  $z^{l-1}$ ,  $z^l$  following a standard normal distribution.

# 2.3 EXPERIMENTS

In the experiments that follow, we utilize the MNIST dataset, a 10-class grayscale image dataset of handwritten digits with 60k training examples and 10k test examples. Then we consider CIFAR-10 (Krizhevsky, 2009), a 10-class color image dataset with 50k training examples and 10k test examples. We use these data to compare our initialization with Xavier and He initializations on a fully connected neural network and with the He initialization on a convolutional neural network.

# 2.3.1 MNIST

Let us verify that our initialization competes with previous weight initialization schemes. To this end, we train a fully connected neural network with GELUs  $(\mu = 0, \sigma = 1)$ , ReLUs, ELUs  $(\alpha = 1)$ , and the tanh activation (Hendrycks & Gimpel, 2016; Clevert et al., 2016). Each 7-layer, 128 neuron wide neural network is trained for 50 epochs with a batch size of 128. We use the Adam optimizer and its suggested learning rate of 0.001 (Kingma & Ba, 2015). We perform this task with no dropout, a dropout keep rate of 0.5, and a dropout keep rate of 0.3. Figure 2 shows that our initialization shows faster convergence at a dropout keep rate of 0.5 for activations like the ReLU and great gains when the dropout keep rate decreases further.

# 2.3.2 CIFAR-10

Since VGG Net architectures (Simonyan & Zisserman, 2015) require considerable regularization and careful initialization, we use a highly regularized variant (Zagoruyko, 2015) of the architecture for our next initialization experiment. The VGG Net-like network has the stacks  $(2 \times 3 \times 64)$ ,  $(2 \times 3 \times 128)$ ,  $(3 \times 3 \times 256)$ ,  $(3 \times 3 \times 512)$ ,  $(3 \times 3 \times 512)$  followed by two fully-connected layers, each with 512 neurons. To regularize the deep network, we keep  $70\%$  of the neurons in the first layer,  $60\%$  in layer 3,  $60\%$  in the first two layers of the last three stacks, and  $50\%$  in the fully connected layers. Max pooling occurs after every stack, ReLU activations are applied on every neuron, and we  $\ell_2$  regularize with a strength of  $5 \times 10^{-4}$ . We compare our initialization (while deactivating the backpropagation variance term) and the He initialization. Since the Xavier initialization is not as prominent in convolutional neural networks we do not test it. We optimize this network with two different optimizers. First, we use Nesterov momentum and tune over the learning rates  $\{10^{-2}, 10^{-3}, 10^{-4}\}$ . In separate runs, we train with the Adam optimizer and tune over the learning rates  $\{10^{-3}, 10^{-4}, 10^{-5}\}$ . With both optimizers we decay the learning rate by 0.1 every at the 100th and 125th epoch all while training for 150 epochs. The results in Figure 3 demonstrate the importance of small corrective dropout factors because the factors' influence on neuron input variance changes exponentially as the network depth increases. We should note that the He initialization rendered the network untrainable when the learning rate was 0.01. However, the network converged with our initialization at this learning rate. Ultimately, our initialization provided consistent, quick, and accurate convergence. With the Adam Optimizer, the VGG Net obtained  $9.51\%$  test set er

![](images/0e6d66194993227864a89f602d1a64f9ebc0f26425bdfa3bef8163e791f4c7c9.jpg)

![](images/6d5de986ca565dcd7a234e573291711f138929a356446489efbfd00188f9e3de.jpg)

![](images/d8c776438f1b53ccf5f663d62cb69d5ae86459e89aa00e509f931f6f27c3920e.jpg)

![](images/f1c1c641036b581b8cf1a09c82513bafff261334196b38b1c2b1101e5578b577.jpg)

![](images/823c4eb04aa0f8b3987756a0978961ab5f767e25a4de5a1a7603d07722134d07.jpg)

![](images/11a8ea87b5bc2e221130b54e2cf43a62504c0cdb7efff9035aadee5fc7b7eb3b.jpg)

![](images/a2e17c5053722c4aa706e734de841fd4ac50d59c91eb6c8eff5c14a337a7dd7c.jpg)

![](images/18dd6ddadbcae7ed59a98ae9d7de9da89db4ec991204f9962524adc5e6476726.jpg)

![](images/02edb08115d615e4ecb7a6277b15c3a64c2f1fd618b2b67db0c6ec50b0f6d143.jpg)

![](images/e00f6243f38592b8b22480d572c1858cc59de462da2ade2ebba963e6ae84add0.jpg)  
Figure 2: MNIST Classification Results. The first row shows the training set log loss curves for the ReLU, the second row is for the tanh unit, the third the ELU, and the fourth the GELU. The leftmost column shows loss curves when there is no dropout, the middle when the dropout rate is 0.5, and rightmost is when the dropout preservation probability rate is 0.3. Each curve is a median of three runs.

![](images/7cf1ac97100c6fa3b72ec0bd88e888d920c90946cff2d9a47177030ddd920f05.jpg)

![](images/bd9ac1b5344388c81c4e24fadd412897f90e12ec86998dd49cc42c6c878653d8.jpg)

ror under our initialization and  $10.54\%$  with the He initialization. Moreover, when using Nesterov momentum, we obtained  $7.61\%$  error with our initialization and  $36.59\%$  with the He initialization.

![](images/63c135249045ab70731a193129a83cdb58a7c98043f213011a5c6a466bc80b16.jpg)  
Figure 3: CIFAR-10 VGG Net Results. The left convergence curves show how the network trained with Nesterov momentum, and the right shows the training under the Adam optimizer. Training set log losses are the darker curves, and the fainter curves are the test set log loss curves.

![](images/5605c53980e90ade0d0e5c6bf1eeb17ba61498e89e9ef6bb187ad2f3c58c1d62.jpg)

# 3 BACH NORMALIZATION VARIANCE RE-ESTIMATION

Batch Normalization aims to prevent an exploding or vanishing feedforward signal just like a good weight initialization. However, Batch Normalization has its own caveats. For example, Mishkin & Matas (2015) claim that Batch Normalization can increase the feedforward time by up to  $30\%$ . Also, Ba et al. (2016) remind us that Batch Normalization cannot be applied to tasks with small batch sizes or online learning tasks lest we normalize a batch based upon mean and variance estimates from a small or single example. Finally, in this section we show that Batch Normalization with dropout also requires special care because the estimated variance Batch Normalization utilizes differs between train and test, suggested by our derivation in section 2.1. In this way, Batch Normalization can be used to stabilize the feedforward signal of a network with dropout during training, removing the need for a weight initialization which corrects for dropout variance. But then Batch Normalization requires that its own variance estimates be corrected before testing. In this section we show how to correct the Batch Normalization variance.

In our experiment, we turn our attention to state of the art convolutional neural networks as they use Batch Normalization and dropout. For example, Densely Connected Networks (DenseNets) (Huang et al., 2016) use dropout with Batch Normalization when training without data augmentation. Training without data augmentation is of interest because it demonstrates how data efficient an architecture is, and some images have their meaning destroyed under augmentation like mirroring (e.g., the mirror image of the digit "7" is meaningless). Moreover, Zagoruyko & Komodakis (2016) use dropout even when there is data augmentation as Batch Normalization alone does not sufficiently regularize the network.

We turn to DenseNets in this experiment because, to our knowledge, they hold the state of the art on CIFAR-10 and CIFAR-100 without data augmentation. We train a DenseNet with dropout and Batch Normalization and re-estimate the Batch Normalization variance parameters outside of training to achieve large error reductions. These DenseNets are trained just as described in the paper except that every 5 epochs we reset the momentum variable following a discussion with one author of the DenseNet paper, as this might improve accuracy. We save the DenseNet model when it has trained for half of the scheduled epochs (when it is "Halfway") and when it is entirely done training. Then, using these models, we feed forward the training data with dropout off for one epoch without performing any backpropagation. While the data feeds forward, we only allow the Batch Normalization moving average estimate of the variance to update. In no way does this variance re-estimation at the Halfway point affect future training because we do not train with these re-estimated variance parameters. Now, DenseNets hold the state of the art on CIFAR-10 and CIFAR-100 without data augmentation. Specifically, they obtain  $5.77\%$  error on CIFAR-10 without data

augmentation and  $23.42\%$  on CIFAR-100 without data augmentation. Table 2 shows the results of Batch Normalization variance moving average re-estimation. The figure shows  $L, k,$  and  $p$ , which are the number of layers  $L$ , the growth factor  $k$ , and the dropout keep probability  $p$ . As an example of a table row, SVHN Original shows the error achieved in the original DenseNet paper. The row below shows the DenseNet we trained at the Halfway point ("Halfway Error") and at the end of training ("Error"), and the error decreased under re-estimating the Batch Normalization variance. The effect of updating the variance estimation is shown under columns with "BN Update." We see that simply feeding forward the training data without dropout and allowing the Batch Normalization variance moving averages to update lets us surpass the state of the art on CIFAR-10 and CIFAR-100 and can sometimes improve accuracy by more than  $2\%$ .

<table><tr><td>Dataset (Architecture)</td><td>Halfway Error</td><td>Halfway Error w/ BN Update</td><td>Error</td><td>Error w/ BN Update</td></tr><tr><td>SVHN (L=40,k=12,p=0.8) Original</td><td>—</td><td>—</td><td>1.79</td><td>—</td></tr><tr><td>SVHN (L=40,k=12,p=0.8) Ours</td><td>5.18</td><td>4.19</td><td>1.92</td><td>1.85</td></tr><tr><td>CIFAR-10 (L=100,k=12,p=0.8) Original</td><td>—</td><td>—</td><td>5.77</td><td>—</td></tr><tr><td>CIFAR-10 (L=100,k=12,p=0.8) Ours</td><td>6.37</td><td>6.07</td><td>5.62</td><td>5.38</td></tr><tr><td>CIFAR-100 (L=100,k=24,p=0.8) Original</td><td>—</td><td>—</td><td>23.42</td><td>—</td></tr><tr><td>CIFAR-100 (L=100,k=12,p=0.8) Original</td><td>—</td><td>—</td><td>23.79</td><td>—</td></tr><tr><td>CIFAR-100 (L=100,k=12,p=0.7) Ours</td><td>24.56</td><td>22.48</td><td>23.91</td><td>22.48</td></tr><tr><td>CIFAR-100 (L=100,k=12,p=0.8) Ours</td><td>23.89</td><td>22.86</td><td>22.65</td><td>22.17</td></tr></table>

Table 2: DenseNet Results with Batch Normalization Variance Re-Estimation. DenseNets without any Batch Normalization variance re-estimation are shown in "Halfway Error" and "Error" columns. Rows with "Original" denote values from Huang et al. (2016). Bold values indicate that the previous state of the art is exceeded. For CIFAR-10 without data augmentation the previous state of the art was  $5.77\%$  and for CIFAR-100 without data augmentation it was  $23.42\%$ .

# 4 DISCUSSION

In practice, if we lack an estimate for a nonlinearity adjustment factor, then 0.5 is a reasonable default. A justification for a 0.5 adjustment factor comes from connections to previous weight initializations. This is because if  $p = 1$  and we default the adjustments to 0.5, our initialization is the "Xavier" initialization if we use vectors from within the unit hypercube rather than vectors on the unit hypersphere (Glorot & Bengio, 2010). Knowing this connection, we can therefore generalize Xavier initialization to

$$
\operatorname {U n i f} [ - 1, 1 ] \times \sqrt {3} / \sqrt {\frac {n _ {\text {i n}}}{p} \mathbb {E} \left[ f \left(z ^ {l - 1}\right) ^ {2} \right] + p n _ {\text {o u t}} \mathbb {E} \left[ f ^ {\prime} \left(z ^ {l}\right) ^ {2} \right]} .
$$

Furthermore, we can optionally exclude the backpropagation variance term—in this case, if  $p = 1$  and  $f$  is a ReLU, our initialization is He's initialization if we use random normal weights (He et al., 2015). Note that since He et al. (2015) considered a 0.5 corrective factor to account for the ReLU's compressiveness (its tendency to reduce output variance), it is plausible that  $\mathbb{E}[f(z^{l - 1})^2]$  is a general adjustment for a nonlinearity's compressiveness. Since most neural network nonlinearities are compressive, 0.5 is a reasonable default adjustment. Also recall that our initialization with the backpropagation variance term amounts is

$$
W ^ {l} / \sqrt {\mathbb {E} [ f (z ^ {l - 1}) ^ {2} ] / p + p \mathbb {E} [ f ^ {\prime} (z ^ {l}) ^ {2} ]}.
$$

If we use the 0.5 corrective factor default and we do not apply any dropout, then we are left with  $W^{l}$ , an orthonormal matrix or a matrix with its columns on a unit hypersphere.

# 5 CONCLUSION

A simple modification to previous weight initializations shows marked improvements on fully connected and convolutional architectures. Unlike recent variance stabilization techniques, ours only relies on simple corrective factors and not special forward passes or batch statistics. For highly-regularized networks, the convergence gains are conspicuous and networks without the corrective factors are harder to train. Therefore, if a user wants to train online or not pay the computational cost Batch Normalization imposes, he or she would do well to apply a dropout corrective factor to their weight initialization matrix.

If a user is able to use Batch Normalization, the effect of dropout still cannot be ignored. The Batch Normalization variance moving averages differ between train and test, so when training is complete, we re-estimate those variance parameters. This is accomplished by feeding the training data forward for one epoch without dropout on and only allowing the variance moving averages to change. By doing so, networks can improve their accuracy notably. Indeed, by applying this simple, highly general technique, we achieved the state of the art on CIFAR-10 and CIFAR-100 without data augmentation.

# ACKNOWLEDGMENTS

We would like to thank Eric Martin for numerous suggestions and Steven Basart for training the SVHN DenseNet. We would also like to thank NVIDIA Corporation for donating several TITAN X GPUs used in this research.

# REFERENCES

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. In arXiv, 2016.  
Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). In International Conference on Learning Representations, 2016.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In International Conference on Artificial Intelligence and Statistics, 2010.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In International Conference on Computer Vision, 2015.  
Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with Gaussian error linear units. In arXiv, 2016.  
Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks. In arXiv, 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.  
Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. International Conference for Learning Representations, 2015.  
Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images, 2009.  
Yann LeCun, Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller. Efficient backprop. In Neural Networks: Tricks of the trade, Springer, 1998.  
Dmytro Mishkin and Jiri Matas. All you need is a good init. In International Conference on Learning Representations, 2015.  
Tim Salimans and Diederik P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Neural Information Processing Systems, 2016.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015.  
Sergey Zagoruyko.  $92.45\%$  on CIFAR-10 in Torch. 2015.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. British Machine Vision Conference, 2016.
