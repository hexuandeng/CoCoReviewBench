# ON BATCH ADAPTIVE TRAINING FOR DEEP LEARNING: LOWER LOSS AND LARGER STEP SIZE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Mini-batch gradient descent and its variants are commonly used in deep learning. The principle of mini-batch gradient descent is to use noisy gradient calculated on a batch to estimate the real gradient, thus balancing the computation cost per iteration and the uncertainty of noisy gradient. However, its batch size is a fixed hyper parameter requiring manual setting before training the neural network. Yin et al. (2017) proposed a batch adaptive stochastic gradient descent (BA-SGD) that can dynamically choose a proper batch size as learning proceeds. We extend the BA-SGD to momentum algorithm, and evaluate both the BA-SGD and the batch adaptive momentum (BA-Momentum) on two deep learning tasks from natural language processing to image classification. Experiments confirm that batch adaptive methods can achieve a lower loss compared with mini-batch methods after scanning the same epochs of data. Furthermore, our BA-Momentum is more robust against larger step sizes, in that it can dynamically enlarge the batch size to reduce the larger uncertainty brought by larger step sizes. The code implementing batch adaptive framework is now open source, applicable to any gradient-based optimization problems.

# 1 INTRODUCTION

Efficiency of training large neural networks becomes increasingly important as deep neural networks tend to have more parameters and require more training data to achieve the state-of-the-art performance on a wide variety of tasks (Goodfellow et al., 2015). For training deep neural networks, stochastic gradient descent (SGD) (Robbins, 2007) and its variants, including momentum, which utilizes past updates with an exponential decay (Qian, 1999), and other methods that can adapt different learning rates for each dimension, such as ADAGRAD (Duchi et al., 2010), ADADELTA (Zeiler, 2012) and ADAM (Kingma & Ba, 2014), are commonly used.

SGD approximates the gradient by only using a single data instance in each iteration, which may lead to uncertainty of approximation. This uncertainty can be reduced by adopting a batch of instances to do the approximation. In mini-batch SGD, the batch size is a fixed hyper parameter requiring manual setting before training the neural network. Setting the batch size typically involves a tuning procedure in which the best batch size is chosen by a series of attempts.

Yin et al. (2017) has developed a batch adaptive stochastic gradient descent (BA-SGD) that can dynamically choose a proper batch size as learning proceeds. BA-SGD models the decrease of objective value as a Gaussian random walk game with rebound on the basis of Taylor extension and central limit theorem. Its core idea is to only update the parameters when the ratio between the expected decrease of objective value and the current batch size is large enough, otherwise enlarge the batch size to better approximate the gradient. It claimed that by smartly choosing the batch size, the BA-SGD not only conserves the fast convergence of SGD algorithm but also avoids too frequent model updates, and compared with mini-batch SGD, its objective value decreases more, after scanning the same amount of data.

However, the experiment in Yin et al. (2017) was only conducted on some simple classification tasks using fully connected neural network with one input layer, one output layer and two hidden layers. What about the evaluation on some complex neural networks, such as convolutional neural network (CNN) and recurrent neural network (RNN)? How well would the batch adaptive algorithm perform on other complicated tasks related to natural language processing and computer vision?

Furthermore, empirical studies reveal that SGD usually performs not so well in some deep and complex neural networks (Sutskever et al., 2013). Can this batch adaptive framework be extended to other gradient based optimization algorithms except SGD?

Therefore, in this paper we extend the batch adaptive framework to momentum algorithm, and evaluate both the batch adaptive SGD (BA-SGD) and the batch adaptive momentum (BA-Momentum) on two deep learning tasks from natural language processing to image classification. These two tasks use RNN and CNN respectively, which cover most of the deep learning models.

In our experiments, we have the following observations. First, for batch adaptive methods, their loss functions converge to lower values after scanning same epochs of data, compared with fixed-batch-size methods. Second, BA-Momentum is more robust against large step sizes by dynamically enlarging the batch size to counteract with the larger noise brought by larger step sizes. Third, we observed a batch size boom, a concentrated period where the batch size frequently increases to larger values, in the training of BA-Momentum. The batch size boom is of significance in that it always appears at the point where mini-batch method starts to reach its lowest possible loss and it helps BA-Momentum keep dropping to even lower loss. More details on these observations and their analysis can be found in Section 4. The code implementing the batch adaptive framework using Theano (AIR) is now open source<sup>1</sup>, which is applicable to any gradient-based optimization problems.

This paper is organized as follows. In Section 2, we briefly introduce the batch adaptive framework proposed by Yin et al. (2017). In Section 3, we extend the batch adaptive framework to momentum algorithm. In Section 4, we demonstrate the performance of BA-M and BA-SGD on Fashion-MNIST (Xiao et al., 2017) and relation extraction task, and then reveal the robustness of BA-Momentum against large step sizes. In Section 5, we discuss some efficiency issue concerned with implementing this batch adaptive framework, and also propose several promising applications based on this framework.

# 2 PREREQUISITES

In this section we briefly summarize the batch adaptive stochastic gradient descent proposed by Yin et al. (2017).

# 2.1 NOTATIONS

We use  $\mathcal{X}$  and  $\mathcal{Y}$  to respectively denote the training data set and its random subset. The vector of model parameters is denoted by  $\vec{\theta}$  and is subscripted by  $t$  to denote an iteration.  $\mathbf{F}$  is used to denote objective function, while  $\mathbf{f}_{\vec{\theta}}$  is the partial derivative of function  $\mathbf{F}$  with model parameters  $\vec{\theta}$  (i.e. the gradient, computed over the whole data set).

# 2.2 BATCH ADAPTIVE FRAMEWORK

Let  $\vec{\xi}_i$  denote the difference between the approximate gradient computed on an individual instance  $\mathbf{g}_{\vec{\theta}}^i$  and the real gradient  $\mathsf{f}_{\vec{\theta}}$ , thus,  $\hat{\mathsf{f}}_{\vec{\theta}}$ , the approximate gradient computed on a batch  $\mathcal{V}$  can be written as:

$$
\hat {\mathbf {f}} _ {\vec {\theta} ^ {\prime}} = \frac {\sum_ {\mathbf {y} _ {j} \in \mathcal {Y}} \mathbf {g} _ {\vec {\theta}} ^ {j}}{| \mathcal {Y} |} = \mathbf {f} _ {\vec {\theta} ^ {\prime}} + \frac {\sum_ {\mathbf {y} _ {j} \in \mathcal {Y}} \vec {\xi} _ {j}}{| \mathcal {Y} |} \tag {1}
$$

The term  $\frac{\sum_{\mathbf{y}_j \in \mathcal{Y}} \vec{\xi}_j}{|\mathcal{Y}|}$  can be viewed as a random variable determined by the randomly sampled batch  $\mathcal{V}$ . On the basis of Central Limit Theorem (CLT), it should satisfy a multi-dimension normal distribution  $\mathcal{N}(\mathbf{0}, \frac{\boldsymbol{\Sigma}}{|\mathcal{V}|})$ , where  $\boldsymbol{\Sigma}$  is the covariance matrix of  $\vec{\xi}_j$ .  $\mathsf{f}_{\vec{\theta}}$  is the real gradient, computed on the whole data set and treated as a constant given the parameters  $\vec{\theta}$ . Thus we have  $\hat{\mathsf{f}}_{\vec{\theta}} \sim \mathcal{N}(\mathsf{f}_{\vec{\theta}}, \frac{\boldsymbol{\Sigma}}{|\mathcal{V}|})$ .

After modelling the estimation of gradient on a batch as a normally distributed random variable, Yin et al. (2017) uses first-order Taylor extension to approximate the objective function  $\mathbf{F}(\vec{\theta})$  at any

parameter configuration. The equation is shown below.

$$
\mathrm {F} (\vec {\theta}) = \mathrm {F} \left(\vec {\theta} _ {0}\right) + \mathrm {f} _ {\vec {\theta} _ {0}} ^ {T} \cdot \left(\vec {\theta} - \vec {\theta} _ {0}\right) + \mathrm {h} _ {\vec {\theta} _ {0}} (\vec {\theta}) \tag {2}
$$

where the function  $\mathfrak{h}_{\vec{\theta}_0}(\vec{\theta})$  is the remainder term satisfying  $\lim_{\vec{\theta}\to \vec{\theta}_0}\mathfrak{h}_{\vec{\theta}_0}(\vec{\theta}) = 0$

If SGD optimization algorithm is adopted to update the parameters, with Equation (1) and Equation (2), the decrease of objective value can be expressed as:

$$
\Delta \mathbf {F} (\vec {\theta} _ {0}) = \mathbf {F} (\vec {\theta} _ {0} - \eta \hat {\mathbf {f}} _ {\vec {\theta} _ {0}}) - \mathbf {F} (\vec {\theta} _ {0}) \approx - \eta \mathbf {f} _ {\vec {\theta} _ {0}} ^ {T} \cdot (\mathbf {f} _ {\vec {\theta} _ {0}} + \frac {\sum_ {\mathbf {y} _ {j} \in \mathcal {Y}} \vec {\xi} _ {j}}{| \mathcal {Y} |}) = - \eta \cdot \mathbf {f} _ {\vec {\theta} _ {0}} ^ {T} \mathbf {f} _ {\vec {\theta} _ {0}} - \eta \cdot \mathbf {f} _ {\vec {\theta} _ {0}} ^ {T} \cdot \vec {\varepsilon} _ {\mathcal {Y}} (3)
$$

where  $\eta$  is the learning rate, and the noise term satisfies  $\vec{\varepsilon}_{\mathcal{Y}} \sim \mathcal{N}(\mathbf{0}, \frac{\mathbf{\Sigma}}{|\mathcal{Y}|})$ , then  $\vec{\varepsilon}_{\mathcal{Y}}^{T} \cdot \mathbf{f}_{\vec{\theta}_{0}}$  can be viewed as a weighted sum of each dimension of vector  $\vec{\varepsilon}_{\mathcal{Y}}$ , thus satisfying a one-dimension Gaussian distribution, i.e.  $\vec{\varepsilon}_{\mathcal{Y}}^{T} \cdot \mathbf{f}_{\vec{\theta}_{0}} \sim \mathcal{N}(0, \frac{\mathbf{f}_{\vec{\theta}_{0}}^{T} \cdot \mathbf{\Sigma} \cdot \mathbf{f}_{\vec{\theta}_{0}}}{|\mathcal{Y}|})$ . Then  $\Delta \mathsf{F}(\vec{\theta}_{0})$  also satisfies a Gaussian distribution, i.e.  $\Delta \mathsf{F}(\vec{\theta}_{0}) \sim \mathcal{N}(-\eta \mathsf{f}_{\vec{\theta}_{0}}^{T} \mathsf{f}_{\vec{\theta}_{0}}, \eta^{2} \frac{\mathbf{f}_{\vec{\theta}_{0}}^{T} \cdot \mathbf{\Sigma} \cdot \mathbf{f}_{\vec{\theta}_{0}}}{|\mathcal{Y}|})$ . In practice, they use the approximate gradient  $\hat{\mathbf{f}}_{\vec{\theta}_{0}}$  to calculate the mean and variance of  $\Delta \mathsf{F}(\vec{\theta}_{0})$ .

For the covariance matrix of  $\vec{\varepsilon}_{\mathcal{Y}}^{\prime}$ ,  $\Sigma$ , its unbiased estimation  $\hat{\Sigma}$  is:

$$
\hat {\boldsymbol {\Sigma}} = \frac {\sum_ {\mathbf {y} _ {j} \in \mathcal {Y}} \left(\mathbf {g} _ {\vec {\theta} _ {0}} ^ {j} - \hat {\mathbf {f}} _ {\vec {\theta} _ {0}}\right) \left(\mathbf {g} _ {\vec {\theta} _ {0}} ^ {j} - \hat {\mathbf {f}} _ {\vec {\theta} _ {0}}\right) ^ {T}}{| \mathcal {Y} | - 1} \tag {4}
$$

After modelling the decrease of objective value as a normally distributed random variable, Yin et al. (2017) abstract the process of objective value change as a random walk game with a Gaussian dice.

In the game, the objective value is regarded as a state, and the decrease of objective value, namely the transfer from the current state to the next state is determined by a Gaussian dice, of which the mean solely depends on the current state and the variance is controlled by the state and the batch to be chosen. They define the domain of game state as a half closed set of real numbers  $[S^{*}, +\infty)$ , where  $S^{*}$  is the minimum objective value that the learning can possibly achieve. The game starts with a random state and the goal is to move as close as possible to  $S^{*}$ . There are two ways of state transfer: one directly decreasing from  $S_{i}$  to  $S_{j}$ , another first reaching minimum point  $S^{*}$  and then rebounding to  $S_{j}$ .

Formally, for state  $s_t$ , the moving step  $\triangle s_t$  is generated by a Gaussian dice  $\mathcal{N}(\mu_t, \frac{\sigma_t^2}{m})$ , where  $\mu_t = \hat{\mathbf{f}}_{\vec{\theta}_t}^T \hat{\mathbf{f}}_{\vec{\theta}_t}$ ,  $\sigma_t^2 = \hat{\mathbf{f}}_{\vec{\theta}_t}^T \hat{\Sigma} \hat{\mathbf{f}}_{\vec{\theta}_t}$ ,  $m = |\mathcal{V}|$ , denoting the batch size. The state transition equation is presented below.

$$
s _ {t + 1} = \left| s _ {t} - S ^ {*} - \eta \bigtriangleup s _ {t} \right| + S ^ {*} = \left\{ \begin{array}{l l} s _ {t} - \eta \bigtriangleup s _ {t}, & \text {i f} s _ {t} - \eta \bigtriangleup s _ {t} \geq S ^ {*} \\ 2 S ^ {*} + \eta \bigtriangleup s _ {t} - s _ {t}, & \text {o t h e r w i s e} \end{array} \right. \tag {5}
$$

Let  $p_m^{s_t}(\triangle s_t)$  denote the probability density function for a random moving step  $\triangle s_t \sim \mathcal{N}\left(\mu_t, \frac{\sigma_t^2}{m}\right)$ . The expected value of next state can be expressed as follows.

$$
\begin{array}{l} \mathbf {E} _ {m} ^ {s _ {t}} \left(s _ {t + 1}\right) = \int_ {- \infty} ^ {+ \infty} p _ {m} ^ {s _ {t}} \left(\triangle s _ {t}\right) \left(\left| s _ {t} - S ^ {*} - \eta \cdot \triangle s _ {t} \right| + S ^ {*}\right) d \triangle s _ {t} \\ = \left(s _ {t} - S ^ {*} - \eta \mu_ {t}\right) \left\{\Phi (a) - \Phi (- a) \right\} + \frac {\eta \sigma_ {t}}{\sqrt {m}} \sqrt {\frac {2}{\pi}} e ^ {- \frac {a ^ {2}}{2}} + S ^ {*} \tag {6} \\ \end{array}
$$

$$
\text {w h e r e} a = \frac {s _ {t} - S ^ {*} - \eta \mu_ {t}}{\eta \sigma_ {t}} \sqrt {m}
$$

To decide the best batch size, one should consider both the variance of estimated gradients and computation cost. Large batch can reduce the variance and therefore make more accurate updates, whereas it also requires more computations. Yin et al. (2017) then define a utility function to find

a balance. Maximizing the utility means achieving the largest expected decrease of loss per data instance.

$$
\mathbf {u} (m, s _ {t}) = \frac {s _ {t} - \mathbf {E} _ {m} ^ {s _ {t}} \left(s _ {t + 1}\right)}{m} \tag {7}
$$

$$
m ^ {*} \leftarrow \arg \max  _ {m} \mathfrak {u} (m, s _ {t}) \tag {8}
$$

where  $m^{*}$  is the best batch size for the  $(t + 1)$ -th iteration.

For more specifications on the BA-SGD algorithm, see Yin et al. (2017).

# 3 BATCH ADAPTIVE MOMENTUM

The main appeal of momentum is its ability to reduce oscillations and accelerate convergence (Goh, 2017). The idea behind momentum is that it accelerates learning along dimensions where gradient continues pointing to the same direction, and slows down those where sign of gradient constantly changes (Zeiler, 2012). Recent work on momentum, called YellowFin, an automatic tuner for both momentum and learning rate, helps momentum optimizer converge in even fewer iterations than ADAM on large ResNet and LSTM models (Zhang et al., 2017). Since it is both powerful and popular, we would like to also apply the batch adaptive framework to momentum optimizer.

# 3.1 MOMENTUM

Momentum utilizes past parameter updates with an exponential decay. Its way of update is given by

$$
\vec {\theta} _ {t + 1} = \vec {\theta} _ {t} - \mathrm {m} _ {t} \tag {9}
$$

$$
\mathfrak {m} _ {t} = \rho \mathfrak {m} _ {t - 1} + \eta \hat {\mathbf {f}} _ {\vec {\theta} _ {t}} \tag {10}
$$

where  $\mathfrak{m}_t$  denotes momentum at the  $t$ -th iteration and  $\rho$  is the rate controlling the decay of the previous parameter updates. Equation 10 can also be written in the following form.

$$
\mathrm {m} _ {t} = \eta \sum_ {\tau = 0} ^ {t} \rho^ {t - \tau} \hat {\mathbf {f}} _ {\vec {\theta} _ {\tau}} \tag {11}
$$

# 3.2 DERIVATION

Referring to Equation 3, we can analogically estimate the decrease of objective value in the following expression.

$$
\begin{array}{l} \Delta \mathbf {F} _ {t} = \mathbf {F} (\vec {\theta} _ {t + 1}) - \mathbf {F} (\vec {\theta} _ {t}) = \mathbf {f} _ {\vec {\theta} _ {t}} ^ {T} (\vec {\theta} _ {t + 1} - \vec {\theta} _ {t}) = - \mathbf {f} _ {\vec {\theta} _ {t}} ^ {T} \mathfrak {m} _ {t} \\ = - \eta \mathbf {f} _ {\vec {\theta} _ {t}} ^ {T} \sum_ {\tau = 0} ^ {t} \rho^ {t - \tau} \hat {\mathbf {f}} _ {\vec {\theta} _ {\tau}} = - \eta \mathbf {f} _ {\vec {\theta} _ {t}} ^ {T} \sum_ {\tau = 0} ^ {t} \rho^ {t - \tau} \left(\mathbf {f} _ {\vec {\theta} _ {\tau}} + \vec {\varepsilon} _ {\tau} ^ {\prime}\right) \tag {12} \\ \end{array}
$$

where  $\vec{\varepsilon}_{\tau} = \frac{\sum_{\mathbf{y}_j \in \mathcal{Y}_{\tau}} \vec{\xi}_j}{|\mathcal{Y}_{\tau}|}$  is the noise term which represents the difference between the real gradient and the estimated gradient calculated on a batch  $\mathcal{Y}_{\tau}$  chosen at the  $\tau$ -th iteration. Though the estimated gradient from the previous iterations (i.e.  $\tau = 0\dots t - 1$ ) has noise, their batches which respectively determine their noise have already been selected, thus their noise is no longer a random variable but a constant. However, for the  $t$ -th iteration, we have not decided which batch to be sampled, therefore  $\vec{\varepsilon}_t$  is indeed a random variable, and on the basis of CLT we know it is normally distributed, i.e.  $\vec{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \frac{\boldsymbol{\Sigma}_t}{|\mathcal{Y}_t|})$ . Based on this and the fact that real gradients,  $\mathsf{f}_{\vec{\theta}_{\tau}}(\tau = 0\dots t)$ , are all constants for the  $t$ -th iteration, we then have the decrease of objective value,  $\Delta \mathbf{F}_t$ , satisfying a one-dimensional Gaussian distribution, which is also experimentally verified in Appendix B.

We need to calculate the mean and variance of  $\Delta \mathrm{F}_t$ , but we prefer not to record all the  $\hat{\hat{\mathbf{F}}}_{\tilde{\theta}_{\tau}}$  from previous iterations. Therefore, we construct a recurrence formula to avoid the trouble.

Let  $\mathcal{P}_t = \sum_{\tau=0}^t \rho^{t-\tau} (\mathfrak{f}_{\vec{\theta}_{\tau}} + \vec{\varepsilon}_{\tau})$ , then we have the following recurrence formula.

$$
\mathcal {P} _ {t} = \left(\mathbf {f} _ {\vec {\theta} _ {t}} + \vec {\varepsilon} _ {t}\right) + \rho \mathcal {P} _ {t - 1} \tag {13}
$$

Thus the mean and variance of  $\mathcal{P}_t$  can be calculated in the following forms.

$$
\mu_ {\mathcal {P} _ {t}} = \rho \mu_ {\mathcal {P} _ {t - 1}} + \hat {\mathbf {f}} _ {\vec {\theta} _ {t}} \tag {14}
$$

$$
\Sigma_ {\mathcal {P} _ {t}} = \frac {\hat {\Sigma} _ {t}}{| \mathcal {Y} _ {t} |} \tag {15}
$$

Now we have  $\Delta \mathbf{F}_t\sim \mathcal{N}(-\eta \hat{\mathbf{f}}_{\vec{\theta}_t}^T\mu_{\mathcal{P}_t},\eta^2\hat{\mathbf{f}}_{\vec{\theta}_t}^T\Sigma_{\mathcal{P}_t}\hat{\mathbf{f}}_{\vec{\theta}_t})$  . In the Gaussian walk game with rebound illustrated in Section 2, the Gaussian dice here satisfies  $\triangle s_t\sim \mathcal{N}(\hat{\mathbf{f}}_{\vec{\theta}_t}^T\mu_{\mathcal{P}_t},\hat{\mathbf{f}}_{\vec{\theta}_t}^T\Sigma_{\mathcal{P}_t}\hat{\mathbf{f}}_{\vec{\theta}_t})$

For simplicity, we let  $\sigma_t^2 = \hat{\mathbf{f}}_{\vec{\theta}_t}^T\hat{\Sigma}_t\hat{\mathbf{f}}_{\vec{\theta}_t}$ ,  $\mu_t = \hat{\mathbf{f}}_{\vec{\theta}_t}^T\mu_{\mathcal{P}_t}$ ,  $m = |\mathcal{V}_t|$ , thus  $\triangle s_t \sim N(\mu_t, \frac{\sigma_t^2}{m})$ . Then the expected value of the next state shares the same expression with Equation 6, though the mean of  $\triangle s_t$  is different.

For batch adaptive momentum algorithm, we also adopt the utility function in Equation 7, and the best batch size is the one that maximizes the utility function, calculated through Equation 8.

# 3.3 ALGORITHM

At the end of this section, we would like to summarize how the batch adaptive momentum algorithm is implemented by presenting the pseudo code below. In the pseudo code, the  $M$  stands for the total budget, i.e. the total number of instances used for training, the  $m_0$  means sampling step, the smallest batch increment. The rest symbols have been introduced before. When implementing the algorithm, one may note that it is time-consuming to calculate the covariance matrix,  $\hat{\Sigma}_t$ . We discuss a tradeoff in Appendix C.

Algorithm 1 Batch-Adaptive Momentum  
1: procedure BA-MOMENTUM  $(\mathcal{X},\vec{\theta},\eta ,\rho ,M,m_0)$    
2: while  $M > 0$  do   
3:  $\mathcal{Y}_t\gets \emptyset$    
4: repeat   
5: random sample  $\mathcal{Z}$  from  $\mathcal{X} - \mathcal{Y}_t$  with  $|\mathcal{Z}| = m_0$    
6:  $\mathcal{Y}_t\gets \mathcal{Y}_t\bigcup \mathcal{Z}$    
7: calculate  $\hat{\mathbf{f}}_{\vec{\theta}_t},\hat{\Sigma}_t$  with  $\mathcal{Y}_t$   Equation (1) and (4)   
8: calculate  $\mu_{\mathcal{P}_t}$  with  $\mu_{\mathcal{P}_{t - 1}}$  and  $\hat{\mathbf{f}}_{\vec{\theta}}$   Equation (14)   
9:  $s_t\gets \mathrm{F}(\vec{\theta} |\mathcal{Y}_t),\mu_t\gets \hat{\mathbf{f}}_{\vec{\theta}_t}^T\mu_{\mathcal{P}_t},\sigma_t\gets \sqrt{\hat{\mathbf{f}}_{\vec{\theta}}^T\hat{\Sigma}_t\hat{\mathbf{f}}_{\vec{\theta}}}$    
10:  $m^{*}\gets \arg \max_{m}\mathrm{u}(m,s_{t})$  where  $S^{*} = 0$    
11: until  $|\mathcal{Y}_t|\geq \min \{m^*,|\mathcal{X}|\}$    
12:  $\mathtt{m}_t = \rho \mathtt{m}_{t - 1} + \eta \hat{\mathbf{f}}_{\vec{\theta}_t}$    
13:  $\vec{\theta}\gets \vec{\theta} -\mathtt{m}_t$    
14:  $M\gets M - |\mathcal{Y}_t|$    
15:  $t = t + 1$    
16: end while   
17: return  $\vec{\theta}$    
18: end procedure

# 4 EXPERIMENT

In this section, we present the learning efficiency of different learning algorithms on two deep learning models. One is the CNN for image classification on the dataset of Fashion MNIST (Xia et al., 2017), and the other is a more complex RNN model for relation extraction on financial

documents (Miwa & Bansal, 2016). To evaluate our learning efficiency regardless of the size of the data set, we use epoch as the unit of scanned data and one epoch means all instances in the whole training set. We calculate the training loss on the whole training set each time when the model has scanned one more epoch of data.

# 4.1 MODEL AND DATASET

Fashion MNIST is a data set of Zalando's article images, which consists of 60000 training samples and 10000 test samples. Each sample is a  $28 \times 28$  grayscale image, associated with a label from 10 classes. We design a CNN model with 3 convolutional layers for this experiment.

Another model we use for relation extraction task is a bi-directional LSTM RNN (Gers, 2001). To train this model, we use 3855 training instances. A detailed description on the specific task and architecture of networks for the two experiments can be found in Appendix A.

# 4.2 LOWER LOSS WITH BUDGET LIMIT

In both experiments, 10 different optimization algorithms are used. They are BA-Momentum, BA-SGD, mini-batch momentum with a fixed size of 32, 64, 128, 256 respectively, and mini-batch SGD with a fixed size of 32, 64, 128, 256 respectively. For simplicity, we denote mini-batch momentum with a fixed size of 32 as "Mini-32-Momentum". The same rule applies to "Mini-256-SGD" etc. We choose mini-batch methods with different sizes ranging from 32 to 256 as baselines. This is because with a budget limit, i.e. fixed number of epochs used for training, the small batch method can have more but less accurate updates, while the large batch method can make fewer but more accurate updates. More updates and accurate updates both can to some extent help the model achieve a lower loss, thus we use both small and large batch methods as baselines to make the comparative test with batch adaptive methods more convincing. For BA-Momentum and BA-SGD, the smallest batch increment,  $m_0$ , is 32, which means the batch size for each iteration in BA-Momentum and BA-SGD is always a multiple 32.

The result of image classification task on Fashion MNIST is plotted in Figure 1 and Figure 3a. The observation shows that for five different momentum-based optimization algorithms, BA-Momentum achieves a much lower loss than the rest four methods after 100 epochs of training. Furthermore, BA-Momentum has fewer fluctuations in later training stage. For the rest four mini-batch methods, the smaller the batch size is, the lower their loss gets and the more they fluctuate. The same trend can be seen in SGD-based methods, where BA-SGD also achieves the lowest loss after a limited budget of 100 epochs.

Now we take a look at the batch size change per iteration, plotted in Figure 3a. For both BA-Momentum and BA-SGD, the batch size keeps almost constant at 32 in an early stage, while it more frequently increases to larger size in later training stage. The tendency is especially evident for BA-SGD. The lines in Figure 3a for BA-SGD gets denser as iterations increase, indicating the batch size rises more frequently from 32 to larger sizes. Also, the largest possible batch size for BA-SGD enlarges from 64 to almost 1000 as learning proceeds.

We display the result of relation extraction task in Figure 2 and Figure 3b. BA-Momentum still achieves the lowest cost compared with other four mini-batch momentum methods. However, BA-SGD almost shares the same curve with Mini-32-SGD, though fluctuates a bit less in later stage. The batch size change for BA-Momentum and BA-SGD on relation extraction task have the same trend observed in the image classification result, though the batch sizes in this task change less frequently, as the sparser lines in Figure 3b suggest.

In terms of the two batch adaptive methods, BA-Momentum and BA-SGD, BA-Momentum shows a faster and lower training loss convergence than BA-SGD in both experiments.

Here we analyze why the BA-Momentum and BA-SGD can achieve the lowest loss on both tasks. The batch adaptive methods adopt a smaller batch in an early stage, which allows them to have more iterations per epoch and thus might accelerate the decrease of loss, then they dynamically enlarge their batch size in later stage to reduce the noise of estimated gradients, because the noise has a larger impact on the accuracy of parameter update in later training stage. Therefore the batch

adaptive methods can conserve the fast decrease of Mini-32 methods in early stage and meanwhile keep decreasing rather than severely fluctuate like Mini-32 methods do in later stage.

![](images/6e265f952b1c566dd3c5f6c96fa44c554278beb8f91f51c2ba10d6a5c98f33cc.jpg)  
(a) Momentum-based methods

![](images/c4982423af06ed0a5eb173b6a19d6b9bb4e24d5e0d0b33475b350a12f10b88af.jpg)  
(b) SGD-based methods

![](images/b48bb6706ad27c7ed9897cf09a5d884a8d38918ebcc88cd5359b521fe9eeb061.jpg)  
Figure 1: Training loss per epoch on Fashion MNIST.  
(a) Momentum-based methods  
Figure 2: Training loss per epoch on relation extraction.

![](images/38281501bfa1762a9a5ee8dc469f9dbee3d06591a35a8ce5b43b048f4af8f339.jpg)  
(b) SGD-based methods

# 4.3 ROBUSTNESS TO LARGE STEP SIZE

In the fine tuning process of BA-Momentum, we find that BA-Momentum is robust against large step sizes. Specifically, under a certain range of learning rates, when we tune the learning rate to a higher value, BA-Momentum can still achieve the same or even lower loss while the performance of other mini-batch methods will be degraded, converging to higher loss and fluctuating more. For simplicity, we only choose Mini-32-Momentum as baseline, since it performs the best in the four momentum-based mini-batch methods. We choose four different learning rates to test the robustness of BA-Momentum. They are 0.005, 0.01, 0.02, and 0.05.

The result is displayed in Figure 4. As shown in Figure 4a, when learning rate rises from 0.005 to 0.01 and then 0.02, BA-Momentum ends up with lower loss from 1.13e-4 to 1.69e-5 and then 5.77e-6. In contrast, Mini-32-Momentum fluctuates more when the learning rate rises from 0.005 to 0.01, and it ends up with much higher loss when the learning rate increases from 0.01 to 0.02. However, when the learning rate is tuned to 0.05, both BA-Momentum and Mini-32-Momentum suffer from a very high loss. These observations confirm that BA-Momentum is more robust against larger step sizes within a certain reasonable range.

![](images/0022cfca8dd3c30de84c2eadc05717bf3d481261b170a23702b212a4f9fb4624.jpg)

![](images/9bbe3b5cdb3476ceadef9ae41d26991b866784c1ce5b85d414cdbede5b4a37c5.jpg)  
(a) Fashion MNIST

![](images/a8428b14a03cb96ea44a4b6240d56fd01480c869302698b23c21ddd7ce5927ef.jpg)

![](images/063e6ed15aabd249fe7e7ebdcf0a48f21e5ef69c6b88e9928b4507bfa599bb98.jpg)  
(b) Relation extraction  
Figure 3: Batch size per iteration on Fashion MNIST and relation extraction.

We offer an explanation on the robustness of BA-Momentum. As the density of lines in Figure 4b suggests, the higher the learning rate is given, the larger our batch size tends to become. BA-Momentum enlarges the batch size to counteract with the larger uncertainty of parameter updates brought by higher learning rate. In this way, it successfully avoids the fluctuations and meanwhile benefits from the larger step size to converge faster.

![](images/1a88d79871a21a462f14e7e336437ba235961aebf70691fa739876b26c54300b.jpg)  
(a) Training loss per epoch  
Figure 4: Effect of different learning rates on BA-Momentum.

![](images/adeeb86d77680413f954004e4248908cb473e824725e51462e7e6d9bdff4fd1f.jpg)  
(b) Batch size per iteration

# 4.4 BATCH SIZE BOOM

We find an intriguing phenomenon when conducting experiments testing the robustness of BA-Momentum. In the process of learning, the batch size of BA-Momentum will experience a batch size boom, a concentrated period where the batch size frequently increases to larger values. For example, the batch size boom in Figure 5a is from 80000 iterations to 110000 iterations, while the batch size boom in Figure 5b is from 60000 iterations to 80000 iterations. In the process of learning, when a batch size boom appears, the curve of BA-Momentum starts to diverge with the curve of Mini-32-Momentum. This makes sense because using a larger batch can help BA-Momentum make more accurate updates and thus decrease to a lower loss. Interestingly, the batch size boom always appears at the point in which Mini-32-Momentum reaches its lowest possible loss and after which it fluctuates around that loss. This can be observed in all three plots in Figure 5. As learning rate

increases, Mini-32-Momentum reaches its lowest loss earlier, and the batch size boom also appears earlier, helping BA-Momentum keep decreasing to lower loss.

![](images/d63624d1e500765bb589a38d081521ea5d8d5fdc7002421faca65fccace56db5.jpg)  
(a) Learning rate  $= 0.005$

![](images/e4da8fdf333886490cea8588b914ab1efdd37246154084b0fd63bb99f09ac1bd.jpg)  
(b) Learning rate  $= 0.01$

![](images/7567ff549f756f65b56b3a2fb6e634a1f6583bc3b3ea6def9a3eff36f659b9ad.jpg)  
(c) Learning rate  $= 0.02$  
Figure 5: Batch size boom on different learning rates.

# 5 CONCLUSION AND DISCUSSION

In this work we developed BA-Momentum algorithm, an extension of the BA-SGD proposed by Yin et al. (2017). We also evaluate the two algorithms on natural language processing and image classification tasks using RNN and CNN respectively. The experiments show that both batch adaptive methods can achieve lower loss than mini-batch methods after scanning same epochs of data. Furthermore, we also confirm that within a certain range of step sizes, BA-Momentum is more robust against large step size compared with mini-batch methods.

In the experiments, we did not evaluate the decrease of training loss with respect to training time. This is because, in the BA-SGD and BA-Momentum algorithm, we have to calculate the derivatives of the loss of each instance from a batch with respect to parameters, and then derive a covariance matrix through Equation 4 from the derivatives. Computing derivatives by backpropagation is time consuming, and now we have to compute all the derivatives of every instance in a batch. However, in mini-batch gradient descent, it is a common practice to calculate an average loss from a batch and then the derivative of this average loss, which requires less time. A feasible approach to reduce the computation cost might be to modify the way Theano do the derivation for a batch of instances and return the square sum of the derivatives, which we plan to study in future work.

The batch adaptive framework can have many important applications. It can be adapted to accelerate distributed deep learning. For distributed deep learning, communication cost for synchronizing gradients and parameters among workers and parameter server is its well-known bottleneck (Li et al., 2014a;b; Wen et al., 2017). A larger batch may help make more accurate updates, thus reducing the total number of iterations needed to converge, lowering the communication cost. However, a larger batch also causes a higher computation cost per iteration. In this update-costly environment, the batch adaptive framework may be modified to take both the computation and communication cost into consideration when deciding a proper batch size, which is worth further exploring.

Another application is that the batch adaptive framework may help remedy the generalization degradation of using large batch studied by Keskar et al. (2016). They provided solid numeric evidence suggesting that using a larger batch will degrade the quality of the model, as measured by its ability to generalize. They also studied the cause for this generalization drop and presented evidence supporting the view that large-batch methods tend to converge to sharp minimizers of the training and testing functions, which causes this generalization drop. Several strategies to help large-batch methods eliminate this generalization drop was proposed in their work. The most promising one is to warm-start with certain epochs of small-batch regime, and then use large batch for the rest of the training. However, the number of epochs needed to warm start with small batch varies for different data sets, thus a batch adaptive method that can dynamically change the batch size against the characteristics of data is the key to solving this problem. The batch adaptive framework sheds light on this issue. Difficulty lies in how to identify a sharp minima accurately and efficiently in the process of learning and limit the batch size when encountering a sharp minima, which we plan to study in future work.

# REFERENCES

Theano: A python framework for fast computation of mathematical expressions.  
John C. Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. In  $COLT$ , 2010.  
Felix Gers. Long short-term memory in recurrent neural networks. Unpublished PhD dissertation, Ecole Polytechnique Fédérale de Lausanne, Lausanne, Switzerland, 2001.  
Gabriel Goh. Why momentum really works. Distill, 2017. doi: 10.23915/distill.00006. URL http://distill.pub/2017/momentum.  
Ian J. Goodfellow, *Yoshua Bengio*, Aaron C. Courville, and Geoffrey E. Hinton. Deep learning. *Scholarpedia*, 10:32832, 2015.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. CoRR, abs/1609.04836, 2016.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.  
Mu Li, David G. Andersen, Alexander J. Smola, and Kai Yu. Communication efficient distributed machine learning with the parameter server. In NIPS, 2014a.  
Mu Li, Tong Zhang, Yuqiang Chen, and Alexander J. Smola. Efficient mini-batch training for stochastic optimization. In KDD, 2014b.  
M. Lichman. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2013.  
Makoto Miwa and Mohit Bansal. End-to-end relation extraction using lstms on sequences and tree structures. CoRR, abs/1601.00770, 2016.  
Ning Qian. On the momentum term in gradient descent learning algorithms. Neural networks, 12 (1):145-151, 1999.  
Herbert Robbins. A stochastic approximation method. 2007.  
Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15:1929-1958, 2014.  
Ilya Sutskever, James Martens, George E. Dahl, and Geoffrey E. Hinton. On the importance of initialization and momentum in deep learning. In ICML, 2013.  
Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. CoRR, abs/1705.07878, 2017.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.  
Peifeng Yin, Ping Luo, and Taiga Nakamura. Small batch or large batch?: Gaussian walk with rebound can teach. In KDD, 2017.  
Matthew D. Zeiler. Adadelta: An adaptive learning rate method. CoRR, abs/1212.5701, 2012.  
Jian Zhang, Ioannis Mitliagkas, and Christopher Ré. Yellowfin and the art of momentum tuning. CoRR, abs/1706.03471, 2017. URL http://arxiv.org/abs/1706.03471.
