# NORMALIZED DIRECTION-PRESERVING ADAM

Anonymous authors

Paper under double-blind review

# ABSTRACT

Optimization algorithms for training deep models not only affects the convergence rate and stability of the training process, but are also highly related to the generalization performance of the models. While adaptive algorithms, such as Adam and RMSprop, have shown better optimization performance than stochastic gradient descent (SGD) in many scenarios, they often lead to worse generalization performance than SGD, when used for training deep neural networks (DNNs). In this work, we identify two problems of Adam that may degrade the generalization performance. As a solution, we propose the normalized direction-preserving Adam (ND-Adam) algorithm, which combines the best of both worlds, i.e., the good optimization performance of Adam, and the good generalization performance of SGD. In addition, we further improve the generalization performance in classification tasks, by using batch-normalized softmax. This study suggests the need for more precise control over the training process of DNNs.

# 1 INTRODUCTION

In contrast with the growing complexity of neural network architectures (Szegedy et al., 2015; He et al., 2016; Hu et al., 2017), the training methods remain relatively simple. Most practical optimization methods for deep neural networks (DNNs) are based on the stochastic gradient descent (SGD) algorithm. However, the learning rate of SGD, as a hyperparameter, is often difficult to tune, since the magnitudes of different parameters can vary widely, and adjustment is required throughout the training process.

To tackle this problem, several adaptive variants of SGD have been developed, including Adagrad (Duchi et al., 2011), Adadelta (Zeiler, 2012), RMSprop (Tieleman & Hinton, 2012), Adam (Kingma & Ba, 2014), etc. These algorithms aim to adapt the learning rate to different parameters automatically, by normalizing the global learning rate based on historical statistics of the gradient w.r.t. each parameter. Although these algorithms can usually simplify learning rate settings, and lead to faster convergence, it is observed that their generalization performance tend to be significantly worse than that of SGD in some scenarios (Wilson et al., 2017). This intriguing phenomenon may explain why SGD (possibly with momentum) is still prevalent in training state-of-the-art deep models, especially feedforward DNNs (Szegedy et al., 2015; He et al., 2016; Hu et al., 2017). Furthermore, recent work has shown that DNNs are capable of fitting noise data (Zhang et al., 2017), suggesting that their generalization capabilities are not the mere result of DNNs themselves, but are entwined with optimization (Arpit et al., 2017).

This work aims to fill the gap between SGD and its adaptive variants. To this end, we identify two problems of Adam that may degrade the generalization performance, and show how these problems are (partially) avoided by using SGD with L2 weight decay. The first problem lies in the fact that the directions of Adam parameter updates are different from that of SGD, i.e., Adam does not preserve the directions of gradients as SGD does. This difference has been discussed in rather recent literature (Wilson et al., 2017), where the authors show that adaptive methods can find drastically different solutions than SGD in some cases. Secondly, while the magnitudes of Adam parameter updates are invariant to rescaling of the gradient, the effect of the updates on the same overall network function still varies with the magnitudes of parameters. As we show, however, this problem can be partially avoided by using SGD with L2 weight decay, which implicitly normalizes the weight vectors, such that the magnitude of each vector's direction change does not depend on its L2-norm.

Next, we propose the normalized direction-preserving Adam (ND-Adam) algorithm, which preserves the direction of the gradient w.r.t. each weight vector, and incorporates a special form of

weight normalization (Salimans & Kingma, 2016). Compared to SGD and Adam, ND-Adam is more robust to improper initialization, and vanishing or exploding gradients. While retaining the superior optimization performance of Adam, ND-Adam achieves the regularization effect of L2 weight decay in a more consistent and principled manner. By using ND-Adam, we are able to achieve significantly better generalization performance than vanilla Adam, and at the same time, obtain much lower training loss at convergence, compared to SGD with L2 weight decay. By closing the gap between SGD and Adam in terms of generalization ability, we also shed some light on why certain optimization algorithms generalize better than others.

Furthermore, we find that the learning signal backpropagated from the softmax layer varies with the overall magnitude of the logits, without proper control. Based on the observation, we apply batch normalization to the logits with a single tunable scaling factor, which further improves the generalization performance in classification tasks.

In essence, our proposed methods, ND-Adam and batch-normalized softmax, enable more precise control over the directions of parameter updates, the learning rates, and the learning signals.

# 2 BACKGROUND AND MOTIVATION

# 2.1 ADAPTIVE MOMENT ESTIMATION (ADAM)

Adaptive moment estimation (Adam) (Kingma & Ba, 2014) is a stochastic optimization method that applies individual adaptive learning rates to different parameters, based on the estimates of the first and second moments of the gradients. Specifically, for  $n$  trainable parameters,  $\theta \in \mathbb{R}^n$ , Adam maintains a running average of the first and second moments of the gradient w.r.t. each parameter as

$$
m _ {t} = \beta_ {1} m _ {t - 1} + (1 - \beta_ {1}) g _ {t}, \tag {1a}
$$

$$
v _ {t} = \beta_ {2} v _ {t - 1} + (1 - \beta_ {2}) g _ {t} ^ {2}. \tag {1b}
$$

Here,  $t$  denotes the time step,  $m_t \in \mathbb{R}^n$  and  $v_t \in \mathbb{R}^n$  denote respectively the first and second moments, and  $\beta_1 \in \mathbb{R}$  and  $\beta_2 \in \mathbb{R}$  are the corresponding decay factors. Kingma & Ba (2014) further notice that, since  $m_0$  and  $v_0$  are initialized to 0's, they are biased towards zero during the initial time steps, especially when the decay factors are large (i.e., close to 1). Thus, for computing the next update, they need to be corrected as

$$
\hat {m} _ {t} = \frac {m _ {t}}{1 - \beta_ {1} ^ {t}}, \hat {v} _ {t} = \frac {v _ {t}}{1 - \beta_ {2} ^ {t}}, \tag {2}
$$

where  $\beta_1^t, \beta_2^t$  are the  $t$ -th powers of  $\beta_1, \beta_2$  respectively. Then, we can update each parameter as

$$
\theta_ {t} = \theta_ {t - 1} - \frac {\alpha_ {t}}{\sqrt {\hat {v} _ {t}} + \epsilon} \hat {m} _ {t}, \tag {3}
$$

where  $\alpha_{t}$  is the global learning rate, and  $\epsilon$  is a small constant to avoid division by zero. Note the above computations between vectors are element-wise.

A distinguishing merit of Adam is that the magnitudes of parameter updates are invariant to rescaling of the gradient, as shown by the adaptive learning rate term,  $\frac{\alpha_t}{\sqrt{\hat{v}_t + \epsilon}}$ . However, there are two potential problems when applying Adam to DNNs.

First, in some scenarios, DNNs trained with Adam generalize worse than that trained with stochastic gradient descent (SGD) (Wilson et al., 2017). Zhang et al. (2017) demonstrate that overparameterized DNNs are capable of memorizing the entire dataset, no matter if it is natural data or meaningless noise data, and thus suggest much of the generalization power of DNNs comes from the training algorithm, e.g., SGD and its variants. It coincides with another recent work (Wilson et al., 2017), which shows that simple SGD often yields better generalization performance than adaptive gradient methods, such as Adam. As pointed out by the latter, the difference in the generalization performance may result from the different directions of updates. Specifically, for each hidden unit, the SGD update of its input weight vector can only lie in the span of all possible input vectors, which, however, is not the case for Adam due to the individually adapted learning rates. We refer to this problem as the direction missing problem.

Second, while batch normalization (Ioffe & Szegedy, 2015) can significantly accelerate the convergence of DNNs, the input weights and the scaling factor of each hidden unit can be scaled in infinitely many (but consistent) ways, without changing the function implemented by the hidden unit. Thus, for different magnitudes of an input weight vector, the updates given by Adam can have different effects on the overall network function, which is undesirable. Furthermore, even when batch normalization is not used, a network using linear rectifiers (e.g., ReLU, leaky ReLU) as activation functions, is still subject to ill-conditioning of the parameterization (Glorot et al., 2011), and hence the same problem. We refer to this problem as the ill-conditioning problem.

# 2.2 L2 WEIGHT DECAY

L2 weight decay is a regularization technique frequently used with SGD. It often has a significant effect on the generalization performance of DNNs. Despite the simplicity and crucial role of L2 weight decay in the training process, it remains to be explained how it works in DNNs. A common justification for L2 weight decay is that it can be introduced by placing a Gaussian prior upon the weights, when the objective is to find the maximum a posteriori (MAP) weights (Blundell et al., 2015). However, as discussed in Sec. 2.1, the magnitudes of input weight vectors are irrelevant in terms of the overall network function, in some common scenarios, rendering the variance of the Gaussian prior meaningless.

We propose to view L2 weight decay in neural networks as a form of weight normalization, which may better explain its effect on the generalization performance. Consider a neural network trained with the following loss function:

$$
\widetilde {L} (\theta ; \mathcal {D}) = L (\theta ; \mathcal {D}) + \frac {\lambda}{2} \sum_ {i \in \mathcal {N}} \| w _ {i} \| _ {2} ^ {2}, \tag {4}
$$

where  $L(\theta; \mathcal{D})$  is the original loss function specified by the task,  $\mathcal{D}$  is a batch of training data,  $\mathcal{N}$  is the set of all hidden units, and  $w_{i}$  denotes the input weights of hidden unit  $i$ , which is included in the trainable parameters,  $\theta$ . For simplicity, we consider SGD updates without momentum. Therefore, the update of  $w_{i}$  at each time step is

$$
\Delta w _ {i} = - \alpha \frac {\partial \widetilde {L}}{\partial w _ {i}} = - \alpha \left(\frac {\partial L}{\partial w _ {i}} + \lambda w _ {i}\right), \tag {5}
$$

where  $\alpha$  is the step size. As we can see from Eq. (5), the gradient magnitude of the L2 penalty is proportional to  $\| w_i\| _2$ , thus forms a negative feedback loop that stabilizes  $\| w_i\| _2$  to an equilibrium value. Empirically, we find that  $\| w_i\| _2$  tends to increase or decrease dramatically at the beginning of the training, and then varies mildly within a small range, which indicates  $\| w_i\| _2\approx \| w_i + \Delta w_i\| _2$ . In practice, we usually have  $\| \Delta w_i\| _2 / \| w_i\| _2\ll 1$ , thus  $\Delta w_{i}$  is approximately orthogonal to  $w_{i}$ , i.e.  $w_{i}\cdot \Delta w_{i}\approx 0$ .

Let  $l_{\parallel w_i}$  and  $l_{\perp w_i}$  be the vector projection and rejection of  $\frac{\partial L}{\partial w_i}$  on  $w_i$ , which are defined as

$$
l _ {\parallel w _ {i}} = \left(\frac {\partial L}{\partial w _ {i}} \cdot \frac {w _ {i}}{\| w _ {i} \| _ {2}}\right) \frac {w _ {i}}{\| w _ {i} \| _ {2}}, l _ {\perp w _ {i}} = \frac {\partial L}{\partial w _ {i}} - l _ {\parallel w _ {i}}. \tag {6}
$$

From Eq. (5) and (6), it is easy to show

$$
\frac {\left\| \Delta w _ {i} \right\| _ {2}}{\left\| w _ {i} \right\| _ {2}} \approx \frac {\left\| l _ {\perp w _ {i}} \right\| _ {2}}{\left\| l _ {\parallel w _ {i}} \right\| _ {2}} \alpha \lambda . \tag {7}
$$

As discussed in Sec. 2.1, when batch normalization is used, or when linear rectifiers are used as activation functions, the magnitude of  $\| w_{i}\|_{2}$  is irrelevant. Thus, it is the direction of  $w_{i}$  that actually makes a difference in the overall network function. If L2 weight decay is not applied, the magnitude of  $w_{i}$ 's direction change will decrease as  $\| w_{i}\|_{2}$  increases during the training process, which can potentially lead to overfitting (discussed in detail in Sec. 3.2). On the other hand, Eq. (7) shows that L2 weight decay implicitly normalizes the weights, such that the magnitude of  $w_{i}$ 's direction change does not depend on  $\| w_{i}\|_{2}$ , and can be tuned by the product of  $\alpha$  and  $\lambda$ . In the following, we refer to  $\| \Delta w_{i}\|_{2} / \| w_{i}\|_{2}$  as the effective learning rate of  $w_{i}$ .

While L2 weight decay produces the normalization effect in an implicit and approximate way, we will show that explicitly doing so can result in further improved optimization and generalization performance.

# 3 NORMALIZED DIRECTION-PRESERVING ADAM

We first present the normalized direction-preserving Adam (ND-Adam) algorithm, which essentially improves the optimization of the input weights of hidden units, while employing the vanilla Adam algorithm to update other parameters. Specifically, we divide the trainable parameters,  $\theta$ , into two sets,  $\theta^v$  and  $\theta^s$ , such that  $\theta^v = \{w_i | i \in \mathcal{N}\}$ , and  $\theta^s = \{\theta \setminus \theta^v\}$ . Then we update  $\theta^v$  and  $\theta^s$  by different rules, as described by Alg. 1. The learning rates for the two sets of parameters are denoted respectively by  $\alpha_t^v$  and  $\alpha_t^s$ .

Algorithm 1: Normalized direction-preserving Adam  
/\* Initialization  $\star /$ $t\gets 0$    
for  $i\in \mathcal{N}$  do   
 $\begin{array}{rl} & w_{i,0}\leftarrow w_{i,0} / \| w_{i,0}\| _2;\\ & m_0(w_i)\leftarrow 0;\\ & v_0(w_i)\leftarrow 0; \end{array}$ $\star$  Perform  $T$  iterations of training  $\star /$    
while  $t <   T$  do   
 $t\gets t + 1$ $\star$  Update  $\theta^v$    
for  $i\in \mathcal{N}$  do   
 $\bar{g}_t(w_i)\gets \partial L / \partial w_i;$ $g_{t}(w_{i})\gets \bar{g}_{t}(w_{i}) - (\bar{g}_{t}(w_{i})\cdot w_{i,t - 1})w_{i,t - 1};$ $m_t(w_i)\gets \beta_1m_{t - 1}(w_i) + (1 - \beta_1)g_t(w_i);$ $v_{t}(w_{i})\gets \beta_{2}v_{t - 1}(w_{i}) + (1 - \beta_{2})\| g_{t}(w_{i})\|_{2}^{2};$ $\hat{m}_t(w_i)\gets m_t(w_i) / (1 - \beta_1^t);$ $\hat{v}_t(w_i)\gets v_t(w_i) / (1 - \beta_2^t);$ $\bar{w}_{i,t}\gets w_{i,t - 1} - \alpha_t^v\hat{m}_t(w_i) / (\sqrt{\hat{v}_t(w_i)} +\epsilon);$ $w_{i,t}\gets \bar{w}_{i,t} / \| \bar{w}_{i,t}\| _2;$    
/\* Update  $\theta^s$  using Adam   
 $\theta_t^s\gets$  AdamUpdate  $(\theta_{t - 1}^s;\alpha_t^s,\beta_1,\beta_2)$    
return  $\theta_T$

In Alg. 1, the iteration over  $\mathcal{N}$  can be performed in parallel, and thus introduces no extra computational complexity. Compared to Adam, computing  $g_{t}(w_{i})$  and  $w_{i,t}$  may take slightly more time, which, however, is negligible in practice. On the other hand, to estimate the second order moment of each  $w_{i} \in \mathbb{R}^{n}$ , Adam maintains  $n$  scalars, whereas ND-Adam requires only one scalar,  $v_{t}(w_{i})$ . Thus, ND-Adam has smaller memory overhead than Adam.

In the following, we address the direction missing problem and the ill-conditioning problem discussed in Sec. 2.1, and explain Alg. 1 in detail. We show how the proposed algorithm jointly solves the two problems, as well as its relation to other normalization schemes.

# 3.1 PRESERVING GRADIENT DIRECTIONS

Assuming the stationarity of a hidden unit's input distribution, the SGD update (possibly with momentum) of the input weight vector is a linear combination of historical gradients, and thus can only lie in the span of the input vectors. As a result, the input weight vector itself will eventually converge to the same subspace.

On the contrary, the Adam algorithm adapts the global learning rate to each scalar parameter independently, such that the gradient of each parameter is normalized by a running average of its magnitudes, which changes the direction of the gradient. To preserve the direction of the gradient w.r.t. each input weight vector, we generalize the learning rate adaptation scheme from scalars to vectors.

Let  $g_{t}(w_{i}), m_{t}(w_{i}), v_{t}(w_{i})$  be the counterparts of  $g_{t}, m_{t}, v_{t}$  for vector  $w_{i}$ . Since Eq. (1a) is a linear combination of historical gradients, it can be extended to vectors without any change; or equivalently, we can rewrite it for each vector as

$$
m _ {t} \left(w _ {i}\right) = \beta_ {1} m _ {t - 1} \left(w _ {i}\right) + \left(1 - \beta_ {1}\right) g _ {t} \left(w _ {i}\right). \tag {8}
$$

We then extend Eq. (1b) as

$$
v _ {t} \left(w _ {i}\right) = \beta_ {2} v _ {t - 1} \left(w _ {i}\right) + \left(1 - \beta_ {2}\right) \| g _ {t} \left(w _ {i}\right) \| _ {2} ^ {2}, \tag {9}
$$

i.e., instead of estimating the average gradient magnitude for each individual parameter, we estimate the average of  $\| g_t(w_i)\| _2^2$  for each vector  $w_{i}$ . In addition, we modify Eq. (2) and (3) accordingly as

$$
\hat {m} _ {t} \left(w _ {i}\right) = \frac {m _ {t} \left(w _ {i}\right)}{1 - \beta_ {1} ^ {t}}, \hat {v} _ {t} \left(w _ {i}\right) = \frac {v _ {t} \left(w _ {i}\right)}{1 - \beta_ {2} ^ {t}}, \tag {10}
$$

and

$$
w _ {i, t} = w _ {i, t - 1} - \frac {\alpha_ {t} ^ {v}}{\sqrt {\hat {v} _ {t} (w _ {i})} + \epsilon} \hat {m} _ {t} (w _ {i}). \tag {11}
$$

Here,  $\hat{m}_t(w_i)$  is a vector with the same dimension as  $w_i$ , whereas  $\hat{v}_t(w_i)$  is a scalar. Therefore, when applying Eq. (11), the direction of the update is the negative direction of  $\hat{m}_t(w_i)$ , and thus is in the span of the historical gradients of  $w_i$ .

It is worth noting that only the input to the first layer (i.e., the training data) is stationary throughout training. Thus, for the weights of an upper layer to converge to the span of its input vectors, it is necessary for the lower layers to converge first. Interestingly, this predicted phenomenon may have been observed in practice (Brock et al., 2017).

Despite the empirical success of SGD, a question remains as to why it is desirable to constrain the input weights in the span of the input vectors. A possible explanation is related to the manifold hypothesis, which suggests that real-world data presented in high dimensional spaces (images, audi- 10s, text, etc) concentrates on manifolds of much lower dimensionality (Cayton, 2005; Narayanan & Mitter, 2010). In fact, commonly used activation functions, such as (leaky) ReLU, sigmoid, tanh, can only be activated (not saturating or having small gradients) by a portion of the input vectors, in whose span the input weights lie upon convergence. Assuming the local linearity of the mani- 10folds of data or hidden-layer representations, constraining the input weights in the subspace that contains some of the input vectors, encourages the hidden units to form local coordinate systems on the corresponding manifold, which can lead to good representations (Rifai et al., 2011).

# 3.2 SPHERICAL WEIGHT OPTIMIZATION

The ill-conditioning problem occurs when the magnitude change of an input weight vector can be compensated by other parameters, such as the scaling factor of batch normalization, or the output weight vector, without affecting the overall network function. Consequently, suppose we have two DNNs that parameterize the same function, but with some of the input weight vectors having different magnitudes, applying the same SGD or Adam update rule will, in general, change the network functions in different ways. Thus, the ill-conditioning problem makes the training process inconsistent and difficult to control.

More importantly, when the weights are not properly regularized (e.g., without using L2 weight decay), the magnitude of  $w_{i}$ 's direction change will decrease as  $\| w_{i}\|_{2}$  increases during the training process. As a result, the effective learning rate for  $w_{i}$  tends to decrease faster than expected, making the network converge to sharp minima (Hoffer et al., 2017). It is well known that sharp minima generalize worse than flat minima (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017).

As shown in Sec. 2.2, L2 weight decay can alleviate the ill-conditioning problem by implicitly and approximately normalizing the weights. However, we still do not have a precise control over  $\| \Delta w_i\| _2 / \| w_i\| _2$ , since  $\| l_{\perp w_i}\| _2 / \| l_{\parallel w_i}\| _2$  is unknown and not necessarily stable. Moreover, the approximation fails when  $\| w_i\| _2$  is far from the equilibrium due to improper initialization, or drastic changes in the magnitudes of the weight vectors. This problem is also addressed by (Neyshabur et al., 2015), by employing a geometry invariant to rescaling of weights. However, their proposed methods do not preserve the direction of gradient.

To address the ill-conditioning problem in a more principled way, we restrict the L2-norm of each  $w_{i}$  to 1, and only optimize its direction. In other words, instead of optimizing  $w_{i}$  in a  $n$ -dimensional space, we optimize  $w_{i}$  on a  $(n - 1)$ -dimensional unit sphere. Specifically, we first obtain the raw gradient w.r.t.  $w_{i}, \bar{g}_{t}(w_{i}) = \partial L / \partial w_{i}$ , and project the gradient onto the unit sphere as

$$
g _ {t} \left(w _ {i}\right) = \bar {g} _ {t} \left(w _ {i}\right) - \left(\bar {g} _ {t} \left(w _ {i}\right) \cdot w _ {i, t - 1}\right) w _ {i, t - 1}. \tag {12}
$$

Here,  $\| w_{i,t - 1}\| _2 = 1$ . Then we follow Eq. (8)-(10), and replace (11) with

$$
\bar {w} _ {i, t} = w _ {i, t - 1} - \frac {\alpha_ {t} ^ {v}}{\sqrt {\hat {v} _ {t} (w _ {i})} + \epsilon} \hat {m} _ {t} (w _ {i}), \tag {13a}
$$

and

$$
w _ {i, t} = \frac {\bar {w} _ {i , t}}{\| \bar {w} _ {i , t} \| _ {2}}. \tag {13b}
$$

In Eq. (12), we keep only the component that is orthogonal to  $w_{i,t-1}$ . However,  $\hat{m}_t(w_i)$  is not necessarily orthogonal as well. In addition, even when  $\hat{m}_t(w_i)$  is orthogonal to  $w_{i,t-1}$ , Eq. (13a) can still increase  $\| w_i \|_2$ , according to the Pythagorean theorem. Therefore, we explicitly normalize  $w_{i,t}$  in Eq. (13b), to ensure  $\| w_{i,t} \|_2 = 1$  after each update. Also note that, since  $w_{i,t-1}$  is a linear combination of its historical gradients,  $g_t(w_i)$  still lies in the span of the historical gradients after the projection in Eq. (12).

As a result, the effective learning rate of a weight vector is

$$
\frac {\left\| \Delta w _ {i , t} \right\| _ {2}}{\left\| w _ {i , t - 1} \right\| _ {2}} \approx \frac {\left\| \hat {m} _ {t} \left(w _ {i}\right) \right\| _ {2}}{\sqrt {\hat {v} _ {t} \left(w _ {i}\right)}} \alpha_ {t} ^ {v}, \tag {14}
$$

which enables precise control over the learning rate of  $w_{i}$  through a single hyperparameter,  $\alpha_{t}^{v}$ , rather than two as required by Eq. (7). Note that it is possible to control the effective learning rate more precisely, by normalizing  $\hat{m}_{t}(w_{i})$  by  $\| \hat{m}_{t}(w_{i})\|_{2}$ , instead of by  $\sqrt{\hat{v}_{t}(w_{i})}$ . However, by doing so, we lose the information provided by  $\| \hat{m}_{t}(w_{i})\|_{2}$  at different time steps. In addition, since  $\hat{m}_{t}(w_{i})$  is less noisy than  $g_{t}(w_{i})$ ,  $\| \hat{m}_{t}(w_{i})\|_{2} / \sqrt{\hat{v}_{t}(w_{i})}$  becomes small near convergence, which is considered a desirable property of Adam (Kingma & Ba, 2014). Thus, we keep the gradient normalization scheme intact.

Compared to SGD with L2 weight decay, spherical weight optimization explicitly normalizes the weight vectors, such that each update to the weight vectors only changes their directions, and strictly keeps the magnitudes constant. Moreover, the magnitude of each update does not depend on the magnitude of the gradient. Thus, ND-Adam is more robust to improper initialization, and vanishing or exploding gradients. For nonlinear activation functions, such as sigmoid and tanh, an extra scaling factor is needed for each hidden unit to express functions that require unnormalized weight vectors. For instance, given an input vector  $x \in \mathbb{R}^n$ , and a nonlinearity  $\phi(\cdot)$ , the activation of hidden unit  $i$  is then given by

$$
y _ {i} = \phi (\gamma_ {i} w _ {i} \cdot x + b _ {i}), \tag {15}
$$

where  $\gamma_{i}$  is the scaling factor, and  $b_{i}$  is the bias.

# 3.3 RELATION TO WEIGHT NORMALIZATION AND BATCH NORMALIZATION

A related normalization and reparameterization scheme, weight normalization (Salimans & Kingma, 2016), has been developed as an alternative to batch normalization, aiming to accelerate the convergence of SGD optimization. We note the difference between spherical weight optimization and weight normalization. First, the weight vector of each hidden unit is not directly normalized in weight normalization, i.e.,  $\| w_{i}\|_{2}\neq 1$  in general. At training time, the activation of hidden unit  $i$  is

$$
y _ {i} = \phi \left(\frac {\gamma_ {i}}{\| w _ {i} \| _ {2}} w _ {i} \cdot x + b _ {i}\right), \tag {16}
$$

which is equivalent to Eq. (15) for the forward pass. For the backward pass,  $\| \Delta w_i\| _2 / \| w_i\| _2$  still depends on  $\| w_{i}\|_{2}$  in weight normalization, hence it does not solve the ill-conditioning problem. At inference time, both of these two schemes can combine  $w_{i}$  and  $\gamma_{i}$  into a single equivalent weight vector,  $w_{i}^{\prime} = \gamma_{i}w_{i}$ , or  $w_{i}^{\prime} = \frac{\gamma_{i}}{\|w_{i}\|_{2}} w_{i}$ .

While spherical weight optimization naturally encompasses weight normalization, it can further benefit from batch normalization. When combined with batch normalization, Eq. (15) evolves into

$$
y _ {i} = \phi \left(\gamma_ {i} \operatorname {B N} \left(w _ {i} \cdot x\right) + b _ {i}\right), \tag {17}
$$

where  $\mathrm{BN}(\cdot)$  represents the transformation done by batch normalization without scaling and shifting. Here,  $\gamma_{i}$  serves as the scaling factor for both the normalized weight vector and batch normalization. At training time, the distribution of the input vector,  $x$ , changes over time, slowing down the training of the sub-network composed by the upper layers. Salimans & Kingma (2016) observe that, such problem cannot be eliminated by normalizing the weight vectors alone, but can be substantially mitigated by combining weight normalization and mean-only batch normalization.

Additionally, in linear rectifier networks, the scaling factors,  $\gamma_{i}$ , can be removed (or set to 1), without changing the overall network function. Since  $w_{i} \cdot x$  is standardized by batch normalization, we have

$$
\mathbb {E} _ {x} \left[ \mathrm {B N} \left(w _ {i} \cdot x\right) ^ {2} \right] \approx 1, \tag {18}
$$

and hence

$$
\operatorname {V a r} _ {x} \left[ \mathrm {B N} \left(w _ {i} \cdot x\right) + b _ {i} \right] \approx 1. \tag {19}
$$

Therefore,  $y_{i}$ 's that belong to the same layer, or different dimensions of  $x$  that fed to the upper layer, will also have comparable variances, which potentially makes the weight updates of the upper layer more stable. For these reasons, we combine the use of spherical weight optimization and batch normalization, as shown in Eq. (17).

# 4 BATCH-NORMALIZED SOFTMAX

For multi-class classification tasks, the softmax function is the de facto activation function for the output layer. Despite its simplicity and intuitive probabilistic interpretation, the learning signal it backpropagates may not always be desirable.

When using cross entropy as the surrogate loss with one-hot target vectors, the prediction is considered correct as long as  $\arg \max_{c\in \mathcal{C}}(z_c)$  is the target class, where  $z_{c}$  is the logit before the softmax activation, corresponding to category  $c\in \mathcal{C}$ . Thus, the logits can be positively scaled together without changing the predictions, even though the cross entropy and its derivatives will vary with the scaling factor. Specifically, denoting the scaling factor by  $\eta$ , the gradient w.r.t. each logit is

$$
\frac {\partial L}{\partial z _ {\hat {c}}} = \eta \left[ \frac {\exp \left(\eta z _ {\hat {c}}\right)}{\sum_ {c \in \mathcal {C}} \exp \left(\eta z _ {c}\right)} - 1 \right], \tag {20a}
$$

and

$$
\frac {\partial L}{\partial z _ {\bar {c}}} = \frac {\eta \exp (\eta z _ {\bar {c}})}{\sum_ {c \in \mathcal {C}} \exp (\eta z _ {c})}. \tag {20b}
$$

where  $\hat{c}$  is the target class, and  $\bar{c} \in \mathcal{C} \setminus \{\hat{c}\}$ .

For Adam and ND-Adam, since the gradient w.r.t. each scalar or vector are normalized, the absolute magnitudes of Eq. (20a) and (20b) are irrelevant. Instead, the relative magnitudes make a difference here. When  $\eta$  is small, we have

$$
\lim  _ {\eta \rightarrow 0} \left| \frac {\partial L / \partial z _ {\bar {c}}}{\partial L / \partial z _ {\hat {c}}} \right| = \frac {1}{| \mathcal {C} | - 1}, \tag {21}
$$

which indicates that, when the magnitude of the logits is small, softmax encourages the logit of the target class to increase, while equally penalizing that of other classes. On the other end of the spectrum, assuming no two digits are the same, we have

$$
\lim  _ {\eta \rightarrow \infty} \left| \frac {\partial L / \partial z _ {\bar {c} ^ {\prime}}}{\partial L / \partial z _ {\hat {c}}} \right| = 1, \lim  _ {\eta \rightarrow \infty} \left| \frac {\partial L / \partial z _ {\bar {c} ^ {\prime \prime}}}{\partial L / \partial z _ {\hat {c}}} \right| = 0, \tag {22}
$$

where  $\vec{c}^{\prime} = \arg \max_{c\in \mathcal{C}\setminus \{\hat{c}\}}(z_c)$ , and  $\vec{c}''\in \mathcal{C}\setminus \{\hat{c},\vec{c} '\}$ . Eq. (22) indicates that, when the magnitude of the logits is large, softmax penalizes only the largest logit of the non-target classes. The latter case is related to the saturation problem of softmax discussed in Oland et al. (2017). However, they focus on the problem of small absolute gradient magnitude, which does not affect Adam and ND-Adam.

It is worth noting that both of these two cases can happen without the scaling factor. For instance, varying the norm of the weights of the softmax layer is equivalent to varying the value of  $\eta$ , in terms of the relative magnitude of the gradient. In the case of small  $\eta$ , the logits of all non-target classes are penalized equally, regardless of the difference in  $\hat{z} - \bar{z}$  for different  $\bar{z} \in \mathcal{C} \setminus \{\hat{z}\}$ . However, it is more reasonable to penalize more the logits that are closer to  $\hat{z}$ , which are more likely to cause misclassification. In the case of large  $\eta$ , although the logit that is most likely to cause misclassification is strongly penalized, the logits of other non-target classes are ignored. As a result, the logits of the non-target classes tend to be similar at convergence, ignoring the fact that some classes are closer to each other than the others.

To exploit the prior knowledge that the magnitude of the logits should not be too small or too large, we apply batch normalization to the logits. Nevertheless, instead of setting  $\gamma_{c}$ 's as trainable variables, we consider them as a single hyperparameter,  $\gamma_{\mathcal{C}}$ , such that  $\gamma_{c} = \gamma_{\mathcal{C}}, \forall c \in \mathcal{C}$ . Tuning the value of  $\gamma_{\mathcal{C}}$  can lead to a better trade-off between the two cases described by Eq. (21) and (22). We refer to this method as batch-normalized softmax (BN-Softmax).

# 5 EXPERIMENTS

In this section, we provide empirical evidence for the analysis in Sec. 2.2, and evaluate the performance of ND-Adam and BN-Softmax on CIFAR-10 and CIFAR-100.

# 5.1 THE EFFECT OF L2 WEIGHT DECAY

To empirically examine the effect of L2 weight decay, we train a wide residual network (WRN) (Zagoruyko & Komodakis, 2016) of 22 layers, with a width of 7.5 times that of a vanilla ResNet. Using the notation in Zagoruyko & Komodakis (2016), we refer to this network as WRN-22-7.5. We train the network on the CIFAR-10 dataset (Krizhevsky & Hinton, 2009), with a small modification to the original WRN architecture, and with a different learning rate annealing schedule. Specifically, for simplicity and slightly better performance, we replace the last fully connected layer with a convolutional layer with 10 output feature maps. I.e., we change the layers after the last residual block from BN-ReLU-GlobalAvgPool-FC-Softmax to BN-ReLU-Conv-GlobalAvgPool-Softmax. In addition, for clearer comparisons, the learning rate is annealed according to a cosine function without restart (Loshchilov & Hutter, 2016; Gastaldi, 2017). We train the model for 80k iterations with a batch size of 128, similar to the settings in Zagoruyko & Komodakis (2016).

As a common practice, we use SGD with a momentum of 0.9, the analysis for which is similar to that in Sec. 2.2. Due to the linearity of derivatives and momentum,  $\Delta w_{i}$  can be decomposed as  $\Delta w_{i} = \Delta w_{i}^{l} + \Delta w_{i}^{p}$ , where  $\Delta w_{i}^{l}$  and  $\Delta w_{i}^{p}$  are the components corresponding to the original loss function,  $L(\cdot)$ , and the L2 penalty term (see Eq. (4)), respectively. Fig. 1a shows the ratio between the scalar projection of  $\Delta w_{i}^{l}$  on  $\Delta w_{i}^{p}$  and  $\| \Delta w_{i}^{p}\|_{2}$ , which indicates how the tendency of  $\Delta w_{i}^{l}$  to increase  $\| w_{i}\|_{2}$  is compensated by  $\Delta w_{i}^{p}$ . Note that  $\Delta w_{i}^{p}$  points to the negative direction of  $w_{i}$ , even when momentum is used, since the direction change of  $w_{i}$  is slow. As shown in Fig. 1a, at the beginning of the training,  $\Delta w_{i}^{p}$  dominates and quickly adjusts  $\| w_{i}\|_{2}$  to its equilibrium value. During the middle stage of the training, the projection of  $\Delta w_{i}^{l}$  on  $\Delta w_{i}^{p}$ , and  $\Delta w_{i}^{p}$  almost cancel each other out. Then, near the end of the training, the gradient of  $w_{i}$  diminishes rapidly to near zero, making  $\Delta w_{i}^{p}$  dominant again. Therefore, Eq. (7) holds more accurately during the middle stage of the training.

In Fig. 1b, we show how the value of  $\| \Delta w_i\| _2 / \| w_i\| _2$  varies in different hyperparameter settings. By Eq. (7),  $\| \Delta w_i\| _2 / \| w_i\| _2$  is expected to remain the same as long as  $\alpha \lambda$  stays constant, which is confirmed by the fact that the curve for  $\alpha_0 = 0.1$ ,  $\lambda = 0.001$  overlaps with that for  $\alpha_0 = 0.05$ ,  $\lambda = 0.002$ . However, comparing the curve for  $\alpha_0 = 0.1$ ,  $\lambda = 0.001$ , with that for  $\alpha_0 = 0.1$ ,  $\lambda = 0.0005$ , we can see that the value of  $\| \Delta w_i\| _2 / \| w_i\| _2$  does not change proportional to  $\alpha \lambda$ . On the other hand, by using ND-Adam, we can control the value of  $\| \Delta w_i\| _2 / \| w_i\| _2$  more precisely by adjusting the learning rate for weight vectors,  $\alpha^v$ . For the same training step, changes in  $\alpha^v$  lead to approximately proportional changes in  $\| \Delta w_i\| _2 / \| w_i\| _2$ , as shown by the two curves corresponding to ND-Adam in Fig. 1b.

Figure 1: An illustration of how L2 weight decay and ND-Adam control the effective learning rate. The results are obtained from the 5th layer of the network, and other layers show similar results.  
![](images/d1cb7eac2381c7f54867094bc872f6e3f61e0d8758e62767474a6e690785d87d.jpg)  
(a) The scalar projection of  $\Delta w_{i}^{l}$  on  $\Delta w_{i}^{p}$  normalized by(b) The relative magnitude of the weight updates, or the  $\| \Delta w_i^p\| _2$  effective learning rate.

![](images/17f82a67623902af151ff1f44567142bee82e44aa2453a9a935ea0f10a47e678.jpg)

# 5.2 PERFORMANCE EVALUATION

To compare the optimization and generalization performance of SGD, Adam, and ND-Adam, we train the same WRN-22-7.5 network on the CIFAR-10 and CIFAR-100 datasets. For SGD and ND-Adam, we first tune the hyperparameters for SGD  $(\alpha_0 = 0.1, \lambda = 0.001$ , momentum 0.9), then tune the initial learning rate of ND-Adam for weight vectors to match the effective learning rate to that of SGD  $(\alpha_0^v = 0.05)$ , as shown in Fig. 1b. While L2 weight decay can greatly affect the performance of SGD, it does not noticeably benefit Adam in our experiments. For Adam and ND-Adam,  $\beta_{1}$  and  $\beta_{2}$  are set to the default values of Adam, i.e.,  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$ . Although the learning rate of Adam is usually set to a constant value, we observe better performance with the cosine decay scheme. The initial learning rate of Adam  $(\alpha_0)$ , and that of ND-Adam for scalar parameters  $(\alpha_0^s)$  are both tuned to 0.001. We use the same data augmentation scheme as used in Zagoruyko & Komodakis (2016), including horizontal flips and random crops, but no dropout is used.

We first experiment with the use of trainable scaling parameters  $(\gamma_{i})$  of batch normalization. As shown in Fig. 2a, ND-Adam converges to training losses comparable to that of Adam, which are much lower than that of SGD. More importantly, as shown in Fig. 2b, the test accuracies of ND-Adam are significantly improved upon vanilla Adam, and matches that of SGD. Note that at the early stage of training, the test accuracy of Adam increases more rapidly than that of ND-Adam and SGD, but remains at a high level afterwards. It is likely that Adam tends to quickly find and get stuck in bad local minima that do not generalize well.

The average results of 3 runs are summarized in the first part of Table 1. Interestingly, compared to SGD, ND-Adam shows slightly better performance on CIFAR-10, but worse performance on CIFAR-100. This inconsistency may be related to the problem of softmax discussed in Sec. 4, that there is a lack of proper control over the magnitude of the logits. But overall, given comparable effective learning rates, ND-Adam and SGD show similar generalization performance. In this sense, the effective learning rate is a more natural learning rate measure than the learning rate hyperparameters.

Next, we repeat the experiments with the use of BN-Softmax. As discussed in Sec. 3.2,  $\gamma_{i}$ 's can be removed from a linear rectifier network, without changing the overall network function. Although this property does not strictly hold for residual networks due to the skip connections, we find that simply removing the scaling factors results in slightly improved generalization performance when using ND-Adam. However, the improvement is not consistent as it degrades performance of SGD. Interestingly, when BN-Softmax is further used, we observe consistent improvement over all three algorithms. Thus, we only report results for this setting.

The scaling factor of the logits,  $\gamma_{\mathcal{C}}$ , is set to 2.5 for CIFAR-10, and 1 for CIFAR-100. As shown in Fig. 3a, the training losses of Adam and ND-Adam again are much lower than that of SGD, although they are increased due to the regularization effect of BN-Softmax. As shown in the second part of

![](images/d8fa9132f8d4a0ce0bc3051cc35af6d1bb3587ecb315d627791f515cd01e270c.jpg)  
(a) The training losses on CIFAR-10/100.

![](images/48c6220c17b197d75c5107f8a3ebb13bfc9479f84edcb240a3f597501aca4e13.jpg)  
(b) The test accuracies on CIFAR-10/100.  
Figure 2: The training losses and test accuracies of the same network trained with SGD, Adam, and ND-Adam. Batch normalization with scaling factors is used.

Table 1, BN-Softmax significantly improves the performance of Adam and ND-Adam. Moreover, in this setting, we obtain the best generalization performance with ND-Adam, outperforming SGD and Adam on both CIFAR-10 and CIFAR-100. As a side note, the optimal value of  $\gamma_{\mathcal{C}}$  tends to remain the same for networks with different widths, but increases for deeper networks. For instance, we observe that  $\gamma_{\mathcal{C}} = 3.5$  works better than  $\gamma_{\mathcal{C}} = 2.5$  for a WRN-34-7.5 network trained on CIFAR-10, although the latter is still better than training without BN-Softmax.

![](images/5b117e1b4e9032f99807f9c46b3047f1ecbddb16ebb8419ae6f012fb56e5b135.jpg)  
(a) The training losses on CIFAR-10/100.

![](images/9955ca7c427c291a53819df2a559783e1496dd860e6844ac6f38ffb6de1c1ab1.jpg)  
(b) The test accuracies on CIFAR-10/100.  
Figure 3: The training losses and test accuracies of the same network trained with SGD, Adam, and ND-Adam. Batch normalization without scaling factors, and BN-Softmax are used.

<table><tr><td>Method</td><td>CIFAR-10 Error (%)</td><td>CIFAR-100 Error (%)</td></tr><tr><td colspan="3">BN w/ scaling factors</td></tr><tr><td>SGD</td><td>4.61</td><td>20.60</td></tr><tr><td>Adam</td><td>6.14</td><td>25.51</td></tr><tr><td>ND-Adam</td><td>4.53</td><td>21.45</td></tr><tr><td colspan="3">BN w/o scaling factors, BN-Softmax</td></tr><tr><td>SGD</td><td>4.49</td><td>20.18</td></tr><tr><td>Adam</td><td>5.43</td><td>22.48</td></tr><tr><td>ND-Adam</td><td>4.14</td><td>19.90</td></tr></table>

Table 1: Test error rates on CIFAR-10 and CIFAR-100.

# 6 CONCLUSION

In this paper, we introduced the normalized direction-preserving Adam algorithm, which is a tailored version of Adam for training DNNs. We showed that ND-Adam preserves the direction of gradient for each weight vector, and implements the regularization effect of L2 weight decay in a more consistent and principled way, such that it combines the good optimization performance of Adam, with the good generalization performance of SGD. In addition, we introduced batch-normalized softmax, which regularizes the logits before the softmax activation, in order to provide better learning signals. We showed significantly improved generalization performance by combining ND-Adam and BN-Softmax. From a high-level view, our analysis and empirical results suggest the need for more precise control over the training process of DNNs.

# REFERENCES

Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxin-der S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. A closer look at memorization in deep networks. arXiv preprint arXiv:1706.05394, 2017.  
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. In ICML 2015, 2015.  
Andrew Brock, Theodore Lim, JM Ritchie, and Nick Weston. Freezeout: Accelerate training by progressively freezing layers. arXiv preprint arXiv:1706.04983, 2017.  
Lawrence Cayton. Algorithms for manifold learning. Univ. of California at San Diego Tech. Rep, pp. 1-17, 2005.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
Xavier Gastaldi. Shake-shake regularization of 3-branch residual networks. In ICLR 2017 Workshop, 2017.  
Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 315-323, 2011.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1-42, 1997.  
Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. arXiv preprint arXiv:1705.08741, 2017.  
Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456, 2015.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. *ICLR* 2017, 2017.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.

Ilya Loshchilov and Frank Hutter. Sgdr: stochastic gradient descent with restarts. arXiv preprint arXiv:1608.03983, 2016.  
Hariharan Narayanan and Sanjoy Mitter. Sample complexity of testing the manifold hypothesis. In Advances in Neural Information Processing Systems, pp. 1786-1794, 2010.  
Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. In Advances in Neural Information Processing Systems, pp. 2422-2430, 2015.  
Anders Oland, Aayush Bansal, Roger B Dannenberg, and Bhiksha Raj. Be careful what you backpropagate: A case for linear output activations & gradient boosting. arXiv preprint arXiv:1707.04199, 2017.  
Salah Rifai, Yann N Dauphin, Pascal Vincent, Yoshua Bengio, and Xavier Muller. The manifold tangent classifier. In Advances in Neural Information Processing Systems, pp. 2294-2302, 2011.  
Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pp. 901-909, 2016.  
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9, 2015.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.  
Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. In Advances in Neural Information Processing Systems, 2017.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Matthew D Zeiler. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701, 2012.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In *ICLR* 2017, 2017.