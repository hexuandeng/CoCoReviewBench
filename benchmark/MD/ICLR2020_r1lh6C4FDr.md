# COMBINED FLEXIBLE ACTIVATION FUNCTIONS FOR DEEP NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Activation in deep neural networks is fundamental to achieving non-linear mappings. Traditional studies mainly focus on finding fixed activations for a particular set of learning tasks or model architectures. The research on flexible activation is quite limited in both designing philosophy and application scenarios. In this study, we propose a general combined form of flexible activation functions as well as three principles of choosing flexible activation component. Based on this, we develop two novel flexible activation functions that can be implemented in LSTM cells and auto-encoder layers. Also two new regularisation terms based on assumptions as prior knowledge are proposed. We find that LSTM and auto-encoder models with proposed flexible activations provides significant improvements on time series forecasting and image compressing tasks, while layer-wise regularization can improve the performance of CNN (LeNet-5) models with RPeLu activation in image classification tasks.

# 1 INTRODUCTION

Deep learning is probably the most powerful technique in modern artificial intelligence (LeCun et al., 2015a). One reason is its ability in approximating complex functions with a large but limited number of parameters (Cybenko, 1989; Hornik, 1991), while the regular layer structures make it possible to be trained with efficient back propagation algorithms (Goodfellow et al., 2016).

In a deep neural network, the weights and bias take account of linear transformation of the data flow, while the activation functions bring in non-linearity. It is remarked in (Hornik, 1991) that activation functions do not perform equally well if we take minimal redundancy or computational efficiency into account. Thus the selection of activation function for different tasks is an issue with importance. Traditionally, people train the weights of linear transformations between layers while keeping the activation functions fixed, and usually one identical activation function is used for all the neurons on each single layer. For example, rectifier linear units (Relu) are used as the default choice for the activation in hidden units for feed forward units and a large proportion of convolutional neural networks (Nair and Hinton, 2010), while sigmoid and tanh functions are used where output values are bounded, such as in output layers for classification problems and the gate activations in recurrent cells (Gers et al., 1999; Chung et al., 2014).

The drawback of Relu activation is the issue of dead unit when the input is negative, which makes people introduce functions with non-zero values in the negative range, including leaky-Relu and Elu (Maas et al., 2013; Clevert et al., 2015). On the other hand, explosion or vanishing gradients in back propagation are also issues that harm the performance of model largely due to the shape of activations (Bengio et al., 1994; Hochreiter, 1998; Pascanu et al., 2012), while techniques such as clipping and batch normalization can be implemented to alleviate these issues to some extent (Ioffe and Szegedy, 2015; Lin et al., 2017).

With a large enough neural network and sufficient training time, the model can effectively learn the patterns from data with possible high accuracy, however it is not straightforward to confirm that learning process is the most efficient and the results are the most accurate. One possible solution for accelerating model training is to introduce flexible or trainable activation functions (Agostinelli et al., 2014; He et al., 2015; Chung et al., 2016). Even though this requires higher computing and storing cost that is proportional to the number of neurons, the performance of non-linear activation function can be largely improved, which could be more efficient than increasing the number of basic model parameters or the number of neurons.

There are existing works trying to promote the predictive performance of deep neural networks based on trainable activation functions. As the Leaky Relu function has a hyper-parameter to be optimized, which is the slope of its negative part, parameterized Relu (PRelu) was proposed to make this slope adapt to the data within specific neurons and be learned during the training process (He et al., 2015). Meanwhile, another study proposes the parameterized version of Elu activation, which introduces two parameters to control the shape of exponential curve in the negative region (Li et al., 2018).

It can also be a blending of different commonly used activations, where the trainable parameters are the weights for the combination components (Sütfeld et al., 2018; Manessi and Rozza, 2018). Since different activation functions can have very similar behavior in some specific regions, a more generative way is to consider their Taylor expansions at 0 point and use a weighted combination of polynomial functions with different orders instead (Chung et al., 2016). For containing those functions that are not centered at 0, one choice is to train a piece-wise function adaptively (Agostinelli et al., 2014). The similar effect can be achieved by Maxout activation, which is quite helpful in promoting the efficiency of models with dropout (Goodfellow et al., 2013). Beyond that, there are also studies on making the most of the non-linear properties by introducing adaptation policy on the input (Flennerhag et al., 2018), which achieve the former state-of-the-art results on several natural language processing (NLP) tasks.

The limitation of existing studies can be illustrated as follows. First, most of existing work focus on some specific forms of parameterized activation functions rather than a more general form. Second, there is a lack of study on flexible versions of bounded activations such as sigmoid and tanh. Third, the experiments of existing work are mainly on convolutional networks rather than other types of architectures such as recurrent networks and auto-encoder networks. In this study, we consider the activation function as a combination of a set of functions following the constraints of several principles. Based on these principles, we develop two novel trainable activation functions that can be introduced to LSTM cells and auto-encoder architecture with significant performance improvement. In addition, layer-wise regularization on activation parameters is introduced to reduce the variance caused by activation functions. Correspondingly, we use three experiments to show the goodness of two novel activation functions and the effect of layer-wise regularization on PReLU activation.

# 2 METHODOLOGY

# 2.1 PARAMERIZED ACTIVATION FUNCTIONS

In this study, we introduce a general form of parameterized activation functions linearly combined from different types of activation functions or a single activation function with multiple parameters. We assume that the parameters in the combined activation functions can be different for each neuron, which can be trained during the main training process of the model parameters with back propagation.

$$
o _ {i} \left(z, \boldsymbol {\alpha} ^ {i}, \boldsymbol {\beta} ^ {i}\right) = \sum_ {k = 1} ^ {K} \alpha_ {i k} a _ {k} \left(z, \boldsymbol {\beta} _ {i k}\right), \quad \sum_ {k = 1} ^ {K} \alpha_ {i, k} = 1 \tag {1}
$$

where  $i$  indexes the neuron, and  $z = z_{l} = W_{l}X_{l - 1} + b_{l}$  is the input of the activation layer indexed by  $l$ . This means that, at each neuron  $i$ , it is possible to have its own set of parameters  $\alpha^i = [\alpha_{i1},\dots,\alpha_{iK}]^T$  and  $\beta^i = [\beta_{i1},\dots,\beta_{iK}]$  where  $\alpha_{ik}$  is the combination weights and  $\beta_{ik}$  is the activation parameter vector for the  $k$ -th component activation  $a_{k}$ , respectively. Thus Eq. (1) defines a form of activation function as a linear combination of a set of basic parameterized non-linear activation functions  $a_{k}(z,\beta_{k})$  with the same input  $x$  to the neuron. Normally, we have  $0\leq \alpha_{i,k}\leq 1$  for all  $k$  and  $i$ . This setting will take advantage of the low computational costs of existing activation functions, while it will be much easier to implement weights normalization when we need a bounded activation function.

Since the specific activation function corresponding to each neuron only depends on its own activation parameters, the back propagation of these activation parameters by stochastic gradient descent can be done as follows:

$$
\alpha_ {i k} \rightarrow \alpha_ {i k} - \gamma \frac {\partial L}{\partial \alpha_ {i k}} \rightarrow \alpha_ {i k} - \gamma \frac {\partial L}{\partial o _ {i}} \cdot \frac {\partial o _ {i}}{\partial \alpha_ {i k}} = \alpha_ {i k} - \gamma \frac {\partial L}{\partial o _ {i}} \cdot a _ {i k} (z, \boldsymbol {\beta} _ {i k}) \tag {2}
$$

$$
\beta_ {i k} \rightarrow \beta_ {i k} - \gamma \frac {\partial L}{\partial \beta_ {i k}} \rightarrow \beta_ {i k} - \gamma \frac {\partial L}{\partial o _ {i}} \cdot \frac {\partial o _ {i}}{\partial \beta_ {i k}} = \beta_ {i k} - \gamma \frac {\partial L}{\partial o _ {i}} \cdot \alpha_ {i k} \frac {\partial a _ {i k} (z , \beta_ {i k})}{\partial \beta_ {i k}}
$$

where  $i$  is the index of the hidden neuron with output  $o_i$  and  $k$  is the index of combined flexible activation functions. Here we use a simplified expression that does not include the indices of layer and training

examples in each mini-batch.  $\gamma$  is the learning rate of gradient descent for all the parameters in activation functions. With the gradients given by  $\partial L / \partial \alpha_{ik}$ , adaptive optimizers such as AdaGrad (Duchi et al., 2011), Adam (Kingma and Ba, 2014) and RMSProp (Tieleman and Hinton, 2017) can also be applied. In general, gradient descent approach and its derivatives can push the activation parameters toward the direction that minimizes the empirical risk of the model on training data. In practice, considering the different nature of basic model parameters such as weights and biases and activation function parameters, it could be more appropriate to implement different learning rates for each of them. However, this will increase the load of hyper-parameter searching.

To build effective combinations with the general form given by Eq. (1), we introduce the following three principles for selecting the components:

- Principle 1: Each component should have the same domain as the baseline activation function.  
- Principle 2: Each component should have a larger or equal range as the baseline activation function.  
- Principle 3: Each component activation functions should be expressively independent of other component functions with the following definition.

Definition 1: If a component activation function  $a_{k}$  is expressively independent of a set of other component functions:  $a_1,\ldots ,a_n$ , there does not exist a set of combination coefficients  $\alpha_{1},\dots,\alpha_{n}$ , inner activation parameters  $\beta_{1},\ldots \beta_{n}$ , parameters of the previous linear layers  $W^{\prime}$ ,  $b^{\prime}$  such that for any input  $X$ , activation parameters  $\beta_{k}$ , and parameters of the previous linear layer  $W_{k}$ ,  $b_{k}$ , the following equation holds:

$$
a _ {k} \left(z _ {k}, \boldsymbol {\beta} _ {\boldsymbol {k}}\right) = a _ {k} \left(\boldsymbol {W} _ {\boldsymbol {k}} X + \boldsymbol {b} _ {k}, \boldsymbol {\beta} _ {\boldsymbol {k}}\right) = \sum_ {i = 1} ^ {n} \alpha_ {i} a _ {i} \left(\boldsymbol {W} ^ {\prime} X + \boldsymbol {b} ^ {\prime}, \boldsymbol {\beta} _ {\boldsymbol {i}}\right) = \sum_ {i = 1} ^ {n} \alpha_ {i} a _ {i} \left(z ^ {\prime}, \boldsymbol {\beta} _ {\boldsymbol {i}}\right) \tag {3}
$$

Theorem 1: For a single-layer network with  $m$  neurons, if a component activation function  $f_{k}$ , which is not expressively independent of other components, is excluded, we need at most  $2m$  neurons to express the same mapping.

The first two principles are aiming at keeping the same ranges and domains of the information flow with the mapping in each layer. The third principle is aiming at reducing the redundant parameters that do not contribute to the model expressiveness even with limited number of units. A short proof of Theorem is provided in Appendix A.1. For example,  $\sigma_{1}(z) = 1 / (1 + e^{-\beta z})$  is not expressively independent with  $\sigma(z) = 1 / (1 + e^{-z})$  since when  $\mathbf{W}' = \beta \mathbf{W}$ , we have  $\sigma(\mathbf{W}'X) = \sigma_{1}(\mathbf{W}X)$ . Therefore, the combined activation  $a(z,\beta) = \alpha_{1}\sigma(z) + (1 - \alpha_{1})\sigma(\beta z)$  will not be a good choice. Based on this, we can then design the combined trainable activation functions for both bounded or unbounded domains.

# 2.2 SIGMOID/TANH FUNCTION EXTENSION FOR RNNS

Sigmoid and Tanh activation functions are widely used in recurrent neural networks, including basic recurrent nets and recurrent nets with cell structure such as LSTMs and GRUs (Gers et al., 1999; Jozefowicz et al., 2015; Goodfellow et al., 2016). For example, an LSTM cell has the functional mapping as follows:

$$
\begin{array}{l} \boldsymbol {f} _ {t} = \sigma \left(\boldsymbol {W} _ {f x} \boldsymbol {x} _ {t} + \boldsymbol {W} _ {f h} \boldsymbol {h} _ {t - 1} + \boldsymbol {b} _ {f}\right) \\ \boldsymbol {i} _ {t} = \sigma \left(\boldsymbol {W} _ {i x} \boldsymbol {x} _ {t} + \boldsymbol {W} _ {i h} \boldsymbol {h} _ {t - 1} + \boldsymbol {b} _ {i}\right) \\ \boldsymbol {o} _ {t} = \sigma \left(\boldsymbol {W} _ {o x} \boldsymbol {x} _ {t} + \boldsymbol {W} _ {o h} \boldsymbol {h} _ {t - 1} + \boldsymbol {b} _ {o}\right) \\ \boldsymbol {g} _ {t} = \tanh  \left(\boldsymbol {W} _ {g x} \boldsymbol {x} _ {t} + \boldsymbol {W} _ {g h} \boldsymbol {h} _ {t - 1} + \boldsymbol {b} _ {g}\right) \tag {4} \\ \boldsymbol {c} _ {t} = \boldsymbol {f} _ {t} * \boldsymbol {c} _ {t - 1} + \boldsymbol {i} _ {t} * \boldsymbol {g} _ {t} \\ \boldsymbol {h} _ {t} = \boldsymbol {o} _ {t} * \operatorname {t a n h} \left(\boldsymbol {c} _ {t}\right) \\ \end{array}
$$

The cell structure includes multiple sigmoid and tanh activation functions, which can be replaced by weighted flexible combination between the original one and another activation function with the same domain. For the sigmoid function, the output should be in the domain of  $[0,1]$ , while for tanh the output should be in  $[-1,1]$ . In the first case, one simple choice is:

$$
o (z; \alpha , \beta) = \alpha \cdot \sigma (z) + (1 - \alpha) \cdot f (z; \beta) \tag {5}
$$

where  $0\leq \alpha \leq 1$  and

$$
f (z; \boldsymbol {\beta}) = \left\{ \begin{array}{l l} 0 & \text {i f} z <   - \frac {1}{2 \beta} \\ \beta z + \frac {1}{2} & \text {i f} - \frac {1}{2 \beta} \leq z \leq \frac {1}{2 \beta} \\ 1 & \text {i f} z > \frac {1}{2 \beta} \end{array} \right. \tag {6}
$$

In Eq. (5),  $f(z; \beta)$  can be considered as a combination of two Ramp functions bounded between 0 and 1 with parameter  $b$ . The shapes of a sample of combined activation with Eq. 5 are shown in Appendix A.3. Similarly, we can build a function with the same boundary as tanh function, and use the corresponding combination to replace tanh in the LSTM cell. Consequently, for each combined flexible activation function, there are two parameters to be optimized during the training process. By combining the original activation function  $\sigma$  with another function  $f(z; \beta)$  with the same boundary using normalized weights, the model can be trained with flexible gates and have the potential to achieve better generalization performance.

# 2.3 RELU FUNCTION EXTENSION FOR MLPS AND CNNS

The outputs of ReLu function is unbounded on positive side, while the derivative with respect to the inputs is a Heaviside step function. To build more flexible activation in the condition when ReLu function is used, we can make a weighted combination between Relu and other non-linear functions with unbounded ends. In the simplest case, we can make a weighted linear combination between ReLu function,  $z^3$  and  $z^{1/3}$ , which can be written as:

$$
o (z; \alpha_ {1}, \alpha_ {2}) = \alpha_ {1} \operatorname {R e L u} (z) + \alpha_ {2} z ^ {3} + (1 - \alpha_ {1} - \alpha_ {2}) z ^ {1 / 3} \tag {7}
$$

where  $\alpha$  and  $\beta$  are two parameters to be learned with back propagation. By merging basic Relu activation and two functions that are expressively independent with other components, such as  $z^3$  and  $z^{1/3}$ , the model could have the potential to learn non-convexity with much less hidden units. In addition, for very deep networks, the combination of ReLu-like function and smooth non-ReLu function could facilitate the information propagation in considering the Edge of Chaos (EOC) (Hayou et al., 2019).

# 2.4 LAYER-WISE REGULARISATION FOR ACTIVATION PARAMETERS

Similar to the weights decay regularisation for model weights in NN models, we introduce regularisation terms for parameters in activation functions to avoid over-parameterization during learning process. When we set the summation of each component's weights in each flexible activation function to 1, it is not suitable to implement a L1 or L2 norm on the absolute value of activation weights. Instead, we use the L2 norm for the absolute difference between each specific activation parameter and the mean of corresponding parameters in the same layer. In addition, we introduce another L2 regularisation term controlling the difference between the trained parameters and the initial parameters of benchmark activation function. This can make sure that the benchmark is actually a specific case of flexible activation, while the variations can be learned to adapt to the training dataset and controlled by these regularisation effects. Thus, the cost function can be written as follows:

$$
\begin{array}{l} \text {C o s t} = \text {P r e d i c t i v e L o s s} + \delta_ {1} \sum_ {j} \frac {\lambda_ {j}}{2 m _ {j}} \sum_ {i} \sum_ {k} | | \alpha_ {i j k} - \bar {\alpha} _ {j k} | | ^ {2} + \delta_ {2} \sum_ {i} \sum_ {j} \sum_ {k} | | \alpha_ {k 0} - \alpha_ {i j k} | | ^ {2} \\ + \text {o t h e r} \\ \end{array}
$$

where  $\alpha_{ijk}$  refers to the  $k$ th activation parameter  $\alpha_{k}$  for  $i$ th element in  $j$ th layer,  $\bar{\alpha}_{jk}$  is the average value of  $\alpha_{k}$  in  $j$ th layer,  $\alpha_{k0}$  is the combination coefficient of  $k$ th component in basic or standard activation functions (e.g. ReLu),  $m_{j}$  is the number of neurons in  $j$ th layer, while  $\lambda_{j}$  is the layer-wise regularisation coefficients, and  $\delta$  is mutual coefficient. We can consider these two regularization terms as priors. For the first one, since in the layer structure of deep neural networks, usually different layers are learning different levels of patterns, which could be in favor of using similar activation functions in each layer. Meanwhile, the second regularization terms can be considered as another priori in assuming that the initial activation functions are good enough and the learned activation parameters should not differ too much from the initial values.

# 3 EXPERIMENTS

# 3.1 EXPERIMENT WITH RECURRENT NEURAL NETWORKS

For testing the performance of the model with flexible activation in recurrent neural networks, we build a multiple-layer LSTM model. We change the three sigmoid functions in Eq. (4) to the parameterized combined function as shown in Eq. (5), then compare the model performances in the cases with or without flexible activations.

The dataset being experimented on is a combination of daily stock returns of five G20 countries including Brazil, Canada, India, China and Japan from 02 Jan, 2009, which is a multi-variate time series data. The five returns of each day can be considered as an input vector to the corresponding hidden unit, while the output is one-step ahead forecast given a sequence of historical data. Instead of using random sampling, we directly split the set of sequences with 30 lagging vectors into training set  $(64\%)$ , validation set  $(16\%)$  and test set  $(20\%)$ , while the learning curve on validation set can be obtained during the training. The loss is selected as the average of mean squared errors of 5 forecasted values with respect to the true values for each example. For the hyper-parameter setting, the batch size was set to be 50, the window size is 10 time steps, the number of epochs is 30, while the optimizer implemented in training is Adam optimizer with the same learning rate on both weights, bias and activation parameters. The initialization of the flexible activation parameters in replacing sigmoid function is  $\alpha = 1$  and  $\beta = 0.1$ , which means that we train them from basement settings. Four stacked LSTM models with different layer configurations are implemented, then we compare the validation and test performances of these models with fixed and flexible activations by 100 trails with different random initializations for each of them.

We introduce the regularizer proposed in Section 2.4 and tune the corresponding weight decay coefficient  $\delta$  for flexible activation parameters. First, for each configuration of layer size, we search for the optimal values of learning rates from 1000 random samples in the range of [0.001, 0.2] with logarithm sale when the fixed activation functions are used. Based on these optimized learning rates for fixed models, we further search for the optimal regularization coefficients for flexible activation functions from the logarithm sale of range [0.001, 0.1] with 30 random samples. The optimal values of the learning rates and the regularization coefficients for activation parameters are listed in Table 1. Here we do provide benefit

Table 1: Summary of hyper-parameter settings for LSTMs in the experiment  

<table><tr><td>Model</td><td>Layer size</td><td>Params</td><td>Learning rates</td><td>Regularization (act)</td></tr><tr><td>LSTM-1</td><td>[5, 16]</td><td>1408</td><td>6.71E-3</td><td>2.50E-1</td></tr><tr><td>LSTM-2</td><td>[5, 8, 4, 4]</td><td>800</td><td>3.47E-3</td><td>2.50E-1</td></tr><tr><td>LSTM-3</td><td>[5, 16, 8]</td><td>2208</td><td>1.93E-2</td><td>2.50E-1</td></tr><tr><td>LSTM-4</td><td>[5, 16, 16]</td><td>3520</td><td>4.32E-3</td><td>2.50E-1</td></tr></table>

to fixed models by using their optimal learning rates in the corresponding flexible models. Meanwhile, to avoid too much extra computational cost paid to flexible models, we use the optimized activation regularization coefficients of LSTM-1 to all the models. Moreover, we use different configurations of layer size to compare models with different capacity to some level.

As is shown in Figure 1, the learning curves of model with flexible activation functions (denoted as "flexible models") on validation set lie below the corresponding curves of models with fixed activation functions (denoted as "fixed models") during most of learning time. Still we consider the average of minimum validation loss during the learning process in each configuration. The descriptive statistical analysis of 100 trails in each setting is shown in Table 2.

Table 2: Summary table of stock indices forecasting with Stacked LSTM  

<table><tr><td>Model</td><td>Data</td><td>Fixed</td><td>Flexible</td><td>Flexible (Regularized)</td></tr><tr><td rowspan="2">LSTM-1</td><td rowspan="2">validation test</td><td>1.747E-4 (1.7E-7)</td><td>1.747E-4 (1.5E-7)</td><td>1.746E-4 (1.4E-7)</td></tr><tr><td>7.954E-5 (8.1E-7)</td><td>7.872E-5 (8.1E-7)</td><td>7.724E-5 (4.8E-7)</td></tr><tr><td rowspan="2">LSTM-2</td><td rowspan="2">validation test</td><td>1.823E-4 (1.9E-7)</td><td>1.804E-4 (2.8E-7)</td><td>1.809E-4 (3.1E-7)</td></tr><tr><td>8.092E-5 (7.7E-7)</td><td>7.811E-5 (5.8E-7)</td><td>7.841E-5 (5.3E-7)</td></tr><tr><td rowspan="2">LSTM-3</td><td rowspan="2">validation test</td><td>1.780E-4 (2.5E-7)</td><td>1.771E-4 (2.3E-7)</td><td>1.770E-4 (2.2E-7)</td></tr><tr><td>8.046E-5 (9.3E-7)</td><td>7.939E-5 (6.2E-7)</td><td>7.794E-5 (5.1E-7)</td></tr><tr><td rowspan="2">LSTM-4</td><td rowspan="2">validation test</td><td>1.771E-4 (2.1E-7)</td><td>1.756E-4 (2.5E-7)</td><td>1.757E-4 (1.9E-7)</td></tr><tr><td>7.925E-5 (7.0E-7)</td><td>7.920E-5 (6.9E-7)</td><td>7.919E-5 (5.9E-7)</td></tr></table>

We can see that in all the configurations of layer size, flexible models outperforms fixed models in terms of minimum validation error and the test error. Especially in the best-performed case with layer size of [5, 16], which is neither the largest or the smallest in terms of the number of parameters, the regularized flexible model achieves an average minimum validation error of 1.746E-04, significantly outperforms all the other models evaluated in this experiment. Further pair-wise statistical tests with normal assumption give p-values between  $10^{-2} \sim 10^{-6}$ . Meanwhile, this model has only  $6.82\%$  more parameters compared with the corresponding fixed model as is calculated in A.3, which is still much smaller than the poorly

![](images/53cf93ded8a1467befc1fe7a01c30f4d60437b0e9762058501a00663211c534f.jpg)  
(a) Time Step [5, 16], window size = 10

![](images/e774224e4599a14f20a4dc18c3392c1e6095529296d6f413990f8668d5a08848.jpg)  
(b) Time Step [5, 8, 4, 4], window size = 10

![](images/97242bd2596e7dbf8535fbc0440f4471365aba8b41250b44824c9f0e439e7805.jpg)  
(c) Time Step [5, 16, 8], window size = 10  
Figure 1: Comparison between the average learning curves (with error bars) of LSTM models with and without regularized flexible activation functions on Multi stock indices return data in forecasting multivariate return. (a) Two-layer LSTM model with layer sizes: [5, 16, 8]; (b) Three-layer LSTM model with layer sizes: [5, 8, 4, 4]; (c) One-layer LSTM model with layer sizes: [5, 16]; (d) Two-layer LSTM model with layer sizes: [5, 16, 16].

![](images/329afd85e2c28fd2d0f9d813787510623b6acfa0b225e9b5aa3697c4552dc691.jpg)  
(d) Time Step [5, 16, 16], window size = 10

performed ones with larger sizes in this experiment. Moreover, this performance improvement can also be observed in further experiments on other combinations of stock indices. Another randomly drawn combination is investigated in A.4.2.

# 3.2 EXPERIMENT WITH DEEP AUTO-ENCODER

The deep auto-encoder based on neural networks is widely implemented in data compression and dimension reduction (Baldi, 2012; Goodfellow et al., 2016). In this experiment, we use two fully connected auto-encoder networks, for which both the encoder and decoder have three hidden layers. The difference between these two basement models is the sizes of two layers in each of the encoder and decoder. The following are the flow graphs of these two models.

$$
\begin{array}{l} \operatorname {I n p u t} (2 8 ^ {*} 2 8) \rightarrow \operatorname {L i n e a r} (2 8 ^ {*} 2 8, 1 2 8) \xrightarrow {\operatorname {R e L u}} \operatorname {L i n e a r} (1 2 8, d _ {1}) \xrightarrow {\operatorname {R e L u}} \operatorname {L i n e a r} (d _ {1}, 1 2) \\ \xrightarrow {\mathrm {R e L u}} \operatorname {L i n e a r} \left(1 2, d _ {2}\right)\rightarrow \operatorname {C o d i n g} \rightarrow \operatorname {L i n e a r} \left(d _ {2}, 1 2\right) \xrightarrow {\mathrm {R e L u}} \operatorname {L i n e a r} \left(1 2, 6 4\right) \xrightarrow {\mathrm {R e L u}} \tag {9} \\ \end{array}
$$

$$
\operatorname {L i n e a r} (6 4, 1 2 8) \xrightarrow {\operatorname {R e L u}} \operatorname {L i n e a r} (1 2 8, 2 8 ^ {*} 2 8) \to \operatorname {O u t p u t} (2 8 ^ {*} 2 8)
$$

Here we use  $d_{1}$  and  $d_{2}$  to denote the layer sizes differ in two models. For the first model "AE1",  $d_{1} = 36$  and  $d_{2} = 6$ , while for the second model "AE2",  $d_{1} = 64$  and  $d_{2} = 3$ . For models with flexible activation, we replace the ReLu activation function with the parameterized function shown in Eq.(9), as well as the existing PReLU activation (He et al., 2015). To avoid adding too many extra parameters, we only introduce flexible activation functions in the 3th and 4th ReLu layers in AE1, while for AE2, they are only introduced in the 2th and 5th ReLu layers. In each trail, we randomly sampled 4,800 training examples, other 1,200 validation examples from 60,000 training examples in the original MNIST dataset, and evaluate the model by the whole 10,000 test examples. The batch size was set to

![](images/112fe308a8c87459b15f3c0dd23a3e76cb4ae06535fb79c664fe17fefad059be.jpg)  
Figure 2: Comparison between the average learning curves (with error bars) of auto-encoder models with and without regularized flexible activation functions on MNIST dataset.

![](images/247cae04e8fa8cb51436b66749aa0c9c648b9bb3ce9af692215daf2e0d9eab78.jpg)

100 and the learning rates of Adam are optimized based on the validation performance of the basement models, which is set to be 0.00645. The training curves are averaged by 50 trails, and the results is demonstrated in Figure 2.

As we can see, for both the two auto-encoder architectures, flexible models out-perform the fixed ones with stable performances. In AE2, the newly proposed activation function significantly outperform PReLU in almost all the epochs, where the flexible activations are added in 2th and 5th ReLu layers. Table 3 gives the corresponding summary for comparing both the validation and test performances of these configurations. It is shown that the test cost and minimum validation cost of flexible models

Table 3: Comparison of auto-encoder models with and without flexible activation functions  

<table><tr><td>Model</td><td>Params</td><td>Data</td><td>ReLU</td><td>PReLU</td><td>Flexible</td></tr><tr><td rowspan="2">AE1</td><td rowspan="2">212,070</td><td rowspan="2">validation test</td><td>3.511E-2 (9.7E-4)</td><td>3.106E-2 (2.6E-4)</td><td>3.120E-2 (1.9E-4)</td></tr><tr><td>3.519E-2 (9.9E-4)</td><td>3.108E-2 (2.7E-4)</td><td>3.122E-2 (1.9E-4)</td></tr><tr><td rowspan="2">AE2</td><td rowspan="2">219,891</td><td rowspan="2">validation test</td><td>3.898E-2 (6.6E-4)</td><td>3.877E-2 (6.6E-5)</td><td>3.708E-2 (7.8E-5)</td></tr><tr><td>3.960E-2 (6.5E-3)</td><td>3.948E-2 (6.9E-5)</td><td>3.770E-2 (6.9E-5)</td></tr></table>

are generally better than that of fixed models with statistical significance in both the two stacked auto-encoder architectures. This advantage can be justified by the results on the test set. The performance of AE1 is generally better since the length of encoded vector is 6 rather than 3 as in AE2. Meanwhile, the flexible models has only less than  $0.03\%$  extra amount of parameters compared with the corresponding fixed ones. Further experiments demonstrated in A.4.3 show that flexible auto-encoder models with 5 encoder layers and 5 decoder layers also outperform fixed models.

# 3.3 EXPERIMENTS WITH CONVOLUTIONAL NEURAL NETWORK (LENET-5)

This experiment investigates the performance of layer-wisely regularized PRelu function in CNNs, it is done with LeNet-5 on CIFAR-10 for image classification (LeCun et al., 2015b; He et al., 2015). The model architecture is shown as follows:

$$
\begin{array}{l} \text {I n p u t} (3 2 * 3 2 * 3) \rightarrow \operatorname {C o n v 2 d} (3, 6, 5) \xrightarrow {\operatorname {R e L u}} \operatorname {M P} (2) \rightarrow \operatorname {C o n v 2 d} (6, 1 6, 5) \xrightarrow {\operatorname {R e L u}} \operatorname {M P} (2) \tag {10} \\ \rightarrow \operatorname {L i n e a r} (1 6 * 5 * 5, 1 2 0) \xrightarrow {\mathrm {B N , R e L u}} \operatorname {L i n e a r} (1 2 0, 8 4) \xrightarrow {\mathrm {B N , R e L u}} \operatorname {L i n e a r} (8 4, 1 0) \rightarrow \operatorname {O u t p u t} \\ \end{array}
$$

where "MP" refers to max-pooling layer, while "BN" refers to batch normalization. In flexible models, we only replace the ReLu activation in the last layer with (6). In each trail, we still randomly sample  $20\%$  of the training set in CIFAR-10 as validation set and use the whole remaining  $80\%$  as the training data. The whole test set in CIFAR-10 are used as the test set in our experiment as well. With a naive random search, the optimized learning rate for fixed model is 0.0012, while the regularization coefficients for the parameter based on this learning rate in flexible is 0.032. Meanwhile, the batch size is still set to be 100.

The average cross-entropy loss on validation set during the training whole time of 10 epochs is given by Figure 3. We can see that the average loss curve of PReLU with layer-wise regularization is almost always below the curve of fixed ReLu and PReLU without regularization, while all the three sets of models are over-fitted after 8th epoch. To check the significance, the corresponding summary statistics for 50 trails is shown in Table 4, where the test results are given by accuracy in classification.

We can see that in Table 4, the average minimal validation cross-entropy loss of regularized flexible models still achieves significant improvement compared with fixed ones (ReLU) and flexible model without regularization (PReLU). Even though the mean accuracy of fixed model seems to be slightly better with batch normalization, it is quite insignificant considering the standard errors. Since the stopping time (10 epochs) is not optimized for both models, this does not make much sense considering the existence of over-fitting. The significant validation results indicate that layer-wise regularization on activation parameters could provide an improvement on models with flexible activation functions even in finely designed benchmark architectures such as LeNet-5.

![](images/134e90b2c6d5648538c8eb68e74e42b723abe5c07a08e32d30f49d03974c63f8.jpg)  
Figure 3: Comparison between the average learning curves (with error bars) of CNN models (LeNet-5) with and without regularized flexible activation functions on CIFAR-10 dataset.

![](images/e23ade2861a83b3edc3fc12c0f0b51a26e8b2dd7ebfc0257502944433cdecdaf.jpg)

Table 4: Comparison of ReLu, PReLu and PReLu with Layer-wise Regularization in LeNet-5  

<table><tr><td>Model</td><td>Params</td><td>Data</td><td>ReLU</td><td>PReLU</td><td>PReLU_reg</td></tr><tr><td rowspan="2">LeNet-5</td><td rowspan="2">61,706</td><td rowspan="2">validation test (acc)</td><td>1.084 (5.7E-3)</td><td>1.0796 (4.6E-3)</td><td>1.075 (4.7E-3)</td></tr><tr><td>62.35 (2.0E-1)</td><td>62.39 (1.7E-1)</td><td>62.67 (1.6E-1)</td></tr><tr><td rowspan="2">LeNet-5 (BN)</td><td rowspan="2">62,522</td><td rowspan="2">validation test (acc)</td><td>1.086 (5.9E-3)</td><td>1.083 (4.9E-3)</td><td>1.072 (4.9E-3)</td></tr><tr><td>62.53 (2.3E-1)</td><td>62.36 (2.0E-1)</td><td>62.48 (1.7E-1)</td></tr></table>

# 4 CONCLUSION

In this study, we proposed a set of principles for designing flexible activation functions in a weighted combination form. Especially, we developed a novel flexible activation function that can be implemented to replace sigmoid and tanh functions in the RNN cells with bounded domains, as well as an alternative one to replace ReLu or PReLu activation in architectures with unbounded domains. In addition, two regularization terms considering the nature of layer-wise feature extraction and goodness of original activation functions are proposed, which is essential in achieving stable improvement of the models. Experiments on multiple time series forecasting show that, with replacing sigmoid activation by the flexible combination proposed in this study, stacked LSTMs can achieve significant improvement. Meanwhile, another proposed flexible combination could significantly improve the performance of auto-encoder networks in image compression. Further experiments indicate that the models with moderately optimized regularized coefficients could also improve the performance of PReLU in CNN (LeNet-5) architectures for image classification. In future studies, it is worthwhile to investigate other flexible activations in combined form based on the proposed framework and principles in this paper, while theoretical justification of the goodness or effectiveness of these activation functions is also a topic of interest.

# REFERENCES

F. Agostinelli, M. Hoffman, P. Sadowski, and P. Baldi. Learning activation functions to improve deep neural networks. arXiv preprint:1412.6830, 2014.  
A. A. Ariyo, A. O. Adewumi, and C. K. Ayo. Stock price prediction using the ARIMA model. In 2014 UKSim-AMSS 16th International Conference on Computer Modelling and Simulation, pages 106-112. IEEE, 2014.  
D. Asteriou and S. G. Hall. ARIMA models and the Box-Jenkins methodology. Applied Econometrics, 2(2):265-286, 2011.  
P. Baldi. Autoencoders, unsupervised learning, and deep architectures. In Proceedings of ICML Workshop on Unsupervised and Transfer Learning, pages 37-49, 2012.  
Y. Bengio, P. Simard, P. Frasconi, et al. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks, 5(2):157-166, 1994.  
H. Chung, S. J. Lee, and J. G. Park. Deep neural network using trainable activation functions. In International Joint Conference on Neural Networks (IJCNN), pages 348-352. IEEE, 2016.  
J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. In NIPS 2014 Workshop on Deep Learning, December 2014, 2014.  
D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint:1511.07289, 2015.  
G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989.  
J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
S. Flennerhag, H. Yin, J. Keane, and M. Elliot. Breaking the activation function bottleneck through adaptive parameterization. In Advances in Neural Information Processing Systems, pages 7739-7750, 2018.  
F. A. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: Continual prediction with LSTM. 1999.  
I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In International Conference on Machine Learning, pages 1319-1327, 2013.  
I. Goodfellow, Y. Bengio, and A. Courville. Deep learning. MIT Press, 2016.  
S. Hayou, A. Doucet, and J. Rousseau. On the impact of the activation function on deep neural networks training. In International Conference on Machine Learning, pages 2672-2680, 2019.  
K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 1026-1034, 2015.  
S. Hochreiter. The vanishing gradient problem during learning recurrent neural nets and problem solutions. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 6(02):107-116, 1998.  
K. Hornik. Approximation capabilities of multilayer feedforward networks. Neural Networks, 4(2): 251-257, 1991.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pages 448-456, 2015.  
R. Jozefowicz, W. Zaremba, and I. Sutskever. An empirical exploration of recurrent network architectures. In International Conference on Machine Learning, pages 2342-2350, 2015.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. Nature, 521(7553):436, 2015a.  
Y. LeCun et al. Lenet-5, convolutional neural networks. URL: http://yann.lecun.com/exdb/lenet, 20:5, 2015b.  
Y. Li, C. Fan, Y. Li, Q. Wu, and Y. Ming. Improving deep neural network with multiple parametric exponential linear units. Neurocomputing, 301:11-24, 2018.  
Y. Lin, S. Han, H. Mao, Y. Wang, and W. J. Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. arXiv preprint:1712.01887, 2017.  
A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In ICML Workshop on Deep Learning for Audio, Speech and Language Processing, 2013.  
F. Manessi and A. Rozza. Learning combinations of activation functions. In 2018 24th International Conference on Pattern Recognition (ICPR), pages 61-66. IEEE, 2018.  
V. Nair and G. E. Hinton. Rectified linear units improve restricted Boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML), pages 807-814, 2010.  
R. Pascanu, T. Mikolov, and Y. Bengio. Understanding the exploding gradient problem. CoRR, abs/1211.5063, 2, 2012.  
L. R. Sutfeld, F. Brieger, H. Finger, S. Fullhase, and G. Pipa. Adaptive blending units: Trainable activation functions for deep neural networks. arXiv preprint:1806.10064, 2018.  
T. Tieleman and G. Hinton. Divide the gradient by a running average of its recent magnitude. coursera: Neural networks for machine learning. Technical Report., 2017.
