# LEARNING TEMPORAL EVOLUTION OF PROBABILITY DISTRIBUTION WITH RECURRENT NEURAL NETWORK

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose to tackle a time series regression problem by computing temporal evolution of a probability density function to provide a probabilistic forecast. A Recurrent Neural Network (RNN) based model is employed to learn a nonlinear operator for temporal evolution of a probability density function. We use a softmax layer for a numerical discretization of a smooth probability density functions, which transforms a function approximation problem to a classification task. Explicit and implicit regularization strategies are introduced to impose a smoothness condition on the estimated probability distribution. A Monte Carlo procedure to compute the temporal evolution of the distribution for a multiple-step forecast is presented. The evaluation of the proposed algorithm on two synthetic and two real data sets shows advantage over the compared baselines.

# 1 INTRODUCTION

Application of the deep learning for manufacturing processes has attracted a great attention as one of the core technologies in Industry 4.0 (Lasi et al., 2014). In many manufacturing processes, e.g. blast furnace, smelter, and milling, the complexity of the overall system makes it almost impossible or impractical to develop a simulation model from the first principles. Hence, system identification from sensor observations has been a long-standing research topic (Wang et al., 2016). Still, when the observation is noisy and there is no prior knowledge on the underlying dynamics, there is only a very limited number of methods for the reconstruction of nonlinear dynamics.

In this work, we consider the following class of problems, where the system is driven by a complex underlying dynamical system, e.g.,

$$
\frac {\partial y}{\partial t} = \mathcal {F} (y (t), y (t - \tau), \boldsymbol {u} (t)). \tag {1}
$$

Here,  $y(t)$  is a continuous process,  $\mathcal{F}$  is a nonlinear operator,  $\tau$  is a delay-time parameter, and  $\pmb{u}(t)$  is an exogenous forcing, such as control parameters. At time step  $t$ , we then observe a noisy measurement of  $y(t)$  which can be defined by the following noise model

$$
\hat {y} _ {t} = y (t) \nu_ {t} + \epsilon_ {t}, \tag {2}
$$

where  $\nu_{t}$  is a multiplicative and  $\epsilon_t$  is an additive noise process. In (1) and (2), we place no assumption on function  $\mathcal{F}$ , do not assume any distributional properties of noises  $\nu_{t}$  and  $\epsilon_{t}$ , but assume the knowledge of the control parameters  $\pmb {u}(t)$ .

Since the noise components,  $\nu_{t}$  and  $\epsilon_{t}$ , are stochastic processes, the observation  $\hat{y}_t$  is a random variable. In this work, we are interested in computing temporal evolution of the probability density function (PDF) of  $\hat{y}$ , given the observations up to time step  $t$ , i.e.,  $p(\hat{y}_{t + n}|\hat{\mathbf{Y}}_{0:t},\mathbf{U}_{0:t + n - 1})$  for  $n\geq 1$ , where  $\widehat{\mathbf{Y}}_{0:t} = (\hat{y}_0,\dots ,\hat{y}_t)$  is a trajectory of the past observations and  $\mathbf{U}_{0:t + n - 1} = (u_0,\dots ,u_{t + n - 1})$  consists of the history of the known control actions,  $\mathbf{U}_{0:t - 1}$ , and a future control scenario,  $\mathbf{U}_{t:t + n - 1}$ . We show, in Section 3, a class of problems, where simple regression problem of forecasting the value of  $\hat{y}_{t + n}$  is not sufficient or not possible, e.g., chaotic systems. Note that the computation of time evolution of a PDF has been a long-standing topic in statistical physics. For a simple Markov process, there are well-established theories based on the Fokker-Planck equation. However, it is very difficult to extend those theories to a more general problem, such as delay-time dynamical systems, or apply it to complex nonlinear systems.

Modeling of the system (1) has been extensively studied in the past, in particular, under the linearity assumptions on  $\mathcal{F}$  and certain noise models, e.g., Gaussian  $\epsilon_{t}$  and  $\nu_{t} = 1$  in (2). The approaches based on auto-regressive processes (Lütkepohl, 2005) and Kalman filter (Harvey, 1990) are good examples. Although these methods do estimate the predictive probability distribution and enable the computation of the forecast uncertainty, the assumptions on the noise and linearity in many cases make it challenging to model real nonlinear dynamical systems.

A recent success of deep learning created a flurry of new approaches for time series modeling and prediction. The ability of deep neural networks, such as RNN, to learn complex nonlinear spatiotemporal relationships in the data enabled these methods to outperform the classical time series approaches. For example, in the recent works of Qin et al. (2017); Hsu (2017); Dasgupta & Osogami (2017), the authors proposed different variants of the RNN-based algorithms to perform time series predictions and showed their advantage over the traditional methods. Although encouraging, these approaches lack the ability to estimate the probability distribution of the predictions since RNN is a deterministic model and unable to fully capture the stochastic nature of the data.

To enable RNN to model the stochastic properties of the data, Chung et al. (2015) augmented RNN with a latent random variable included in the hidden state and proposed to estimate the resulting model using variational inference. In a similar vein, the works of Archer et al. (2015); Krishnan et al. (2017) extend the traditional Kalman filter to handle nonlinear dynamics when the inference becomes intractable. Their approach is based on formulating the variational lower bound and optimizing it under the assumption of Gaussian posterior.

Another recent line of works enabled stochasticity in the RNN-based models by drawing a connection between Bayesian variation inference and a dropout technique. In particular, Gal & Ghahramani (2016) showed that the model parameter uncertainty (which then leads to uncertainty in model predictions), that traditionally was estimated using variational inference, can be approximated using a dropout method (a random removal of some connections in the network structure). The prediction uncertainty is then estimated by evaluating the model outputs at different realizations of the dropout weights. Following the ideas of Gal & Ghahramani (2016), Zhu & Laptev (2017) proposed additional ways (besides modeling the parameter uncertainty) to quantify the forecast uncertainty in RNN, which included the model mis-specification error and the inherent noise of the data.

# 1.1 OVERVIEW OF THE PROPOSED WORK

We propose an RNN-model to compute the temporal evolution of a PDF,  $p(\hat{y}_{t + n}|\widehat{\mathbf{Y}}_{0:t},\mathbf{U}_{0:t + n - 1})$ . To avoid the difficulties in directly estimating the continuous function, we use a numerical discretization technique, which converts the function approximation problem to a classification task (see Section 2.2). We note that the use of the traditional cross-entropy (CE) loss in our formulated classification problem can be problematic since it is oblivious to the class ordering. To address this, we additionally propose two regularizations for CE to account for a geometric proximity between the classes (see Sections 2.2.1 and 2.2.2). The probability distribution of one-step-ahead prediction,  $p(\hat{y}_{t + 1}|\widehat{\mathbf{Y}}_{0:t},\mathbf{U}_{0:t})$  can now be simply estimated from the output softmax layer of RNN (see Section 2.2), while to propagate the probability distribution further in time, for a multiple-step forecast, we propose a sequential Monte Carlo (SMC) method (see Section 2.3). We empirically show that the proposed modeling approach enables us to represent a continuous PDF of any arbitrary shape, including the ability to handle the multiplicative data noises in (2). Since the probability distribution is computed, the RNN-model can also be used for a regression task by computing the expectation (see Section 2.3). Hereafter, we use DE-RNN for the proposed RNN model, considering the similarity with the density-estimation task.

In summary, the contributions of this work are as follows: (i) formulate the classical regression problem for time series prediction as a predictive density-estimation problem, which can be solved by a classification task (ii) propose an approach to compute the time evolution of probability distribution using SMC on the distributions from DE-RNN (iii) proposed two regularizations for CE loss to capture the ordering of the classes in the discretized PDF. We evaluated the proposed algorithm on two synthetic and two real datasets, showing its advantage over the baselines. Note that DE-RNN has a direct relevance to a wide range of problems in physics and engineering, in particular, for uncertainty quantification and propagation (Zhang & Karniadakis, 2017).

# 2 LSTM FOR NOISY DYNAMICAL SYSTEM

In this Section we present the details of the proposed algorithm using a specific form of RNN, called Long Short-Term Memory (LSTM) network. Although in the following presentation and experiments we used LSTM, other networks, e.g., GRU (Chung et al., 2014), can be used instead.

# 2.1 REVIEW OF LONG SHORT-TERM MEMORY NETWORK

The Long Short-Term Memory network (LSTM) (Hochreiter & Schmidhuber, 1997; Gers et al., 2000) consists of a set of nonlinear transformations of input variables  $\mathbf{z}_t \in \mathbb{R}^m$ ;

$$
G a t i n g \text {f u n c t i o n s}: G _ {i, f, o} = \varphi_ {S} \circ \mathcal {L} \left(\boldsymbol {z} _ {t}\right), \tag {3}
$$

$$
\text {I n t e r n a l} \quad s _ {t} = \left(\mathbf {1} - \boldsymbol {G} _ {f}\right) \odot s _ {t - 1} + \boldsymbol {G} _ {i} \odot \left(\varphi_ {T} \circ \mathcal {L} \left(\boldsymbol {z} _ {t}\right)\right), \tag {4}
$$

$$
\text {O u t p u t}: \boldsymbol {h} _ {t} = \boldsymbol {G} _ {o} \odot \boldsymbol {s} _ {t}. \tag {5}
$$

Here,  $\varphi_S$  and  $\varphi_T$ , respectively, denote the sigmoid and hyperbolic tangent functions,  $\mathcal{L}$  is a linear layer, which includes a bias,  $s_t \in \mathbb{R}^{N_c}$  is the internal state,  $h_t \in \mathbb{R}^{N_c}$  is the output of the LSTM network,  $N_c$  is the number of the LSTM units, and  $a \odot b$  denote a component-wise multiplication.

Interesting observation can be made about equation (4). We can re-write equation (4) as

$$
\boldsymbol {s} _ {t + 1} = \left[ 1 - f \left(\boldsymbol {z} _ {t}\right) \right] \boldsymbol {s} _ {t} + g \left(\boldsymbol {z} _ {t}\right), \tag {6}
$$

for some functions  $f$  and  $g$ . With a simple re-scaling, this equation can be interpreted as a first-order Euler scheme for a linear dynamical system,

$$
\frac {d \boldsymbol {s}}{d t} = - f (\boldsymbol {z}) \boldsymbol {s} + g (\boldsymbol {z}). \tag {7}
$$

Thus, LSTM can be understood as a series expansion, where a complex nonlinear dynamical system is approximated by a combination of many simpler dynamical systems.

Usually, LSTM network is supplemented by feed-forward neural networks, e.g.,

$$
\boldsymbol {z} _ {t} = \mathcal {F} _ {\text {i n}} \left(\boldsymbol {x} _ {t}, \boldsymbol {h} _ {t - 1}\right), \quad \boldsymbol {P} _ {t + 1} = \mathcal {F} _ {\text {o u t}} \left(\boldsymbol {h} _ {t}\right), \tag {8}
$$

in which  $x_{t}$  is the input feature. Using (5), we can denote by  $\Psi_{e}$  and  $\Psi_{d}$  a collection of the operators from input to internal state (encoder) and from internal state to the output  $o$  (decoder):

$$
\boldsymbol {s} _ {t} = \Psi_ {e} \left(\boldsymbol {x} _ {t}, \boldsymbol {s} _ {t - 1}\right), \quad \boldsymbol {P} _ {t + 1} = \Psi_ {d} \left(\boldsymbol {s} _ {t}\right). \tag {9}
$$

# 2.2 DISCRETE APPROXIMATION OF PROBABILITY DENSITY FUNCTION

In this Section we first consider the problem of modeling the conditional PDF,  $p(\hat{y}_{t+1}|\widehat{\mathbf{Y}}_{0:t}, \mathbf{U}_{0:t})$ . Although  $\hat{y}_{t+1}$  has a dependence on the past trajectories of both  $\hat{y}$  and  $\mathbf{u}$ , using the "state space" LSTM model argument in Section 2.1, the conditional PDF can be modeled as a Markov process

$$
p \left(\hat {y} _ {t + 1} \mid \widehat {\mathbf {Y}} _ {0: t}, \boldsymbol {U} _ {0: t}\right) = p \left(\hat {y} _ {t + 1} \mid \hat {y} _ {t}, \boldsymbol {u} _ {t}, \boldsymbol {s} _ {t - 1}\right) = p \left(\hat {y} _ {t + 1} \mid \boldsymbol {s} _ {t}\right). \tag {10}
$$

Hence, to simplify the problem, we consider a task of estimating the PDF of a random variable  $\hat{y}$ , given an input  $x$ , i.e.,  $p(\hat{y} | x)$ . The obtained results can then be directly applied to the original problem of estimating  $p(\hat{y}_{t + 1} | s_t)$ .

Let  $\alpha = (\alpha_{0},\dots ,\alpha_{K})$  denote a set of real numbers, such that  $\alpha_{i - 1} < \alpha_{i}$  for  $i = 1,\dots ,K$ , which defines  $K$  disjoint intervals,  $\mathcal{I}_i = (\alpha_{i - 1},\alpha_i)$ . Then, a discrete probability distribution can be defined

$$
p (k | x) = \int_ {\mathcal {I} _ {k}} p (\hat {y} | x) d y, \text {f o r} k = 1, \dots , K, \tag {11}
$$

where it is clear that  $p(k|x)$  is a numerical discretization of the continuous PDF,  $p(\hat{y} |x)$ . Using the LSTM from Section 2.1, the discrete probability  $p(k|x)$  can be modeled by the softmax layer  $(P)$  as an output of  $\Psi_d$  in (9) such that

$$
p (k | x) = P _ {k}, \text {f o r} k = 1, \dots , K. \tag {12}
$$

Thus, the original problem of estimating a smooth function,  $p(\hat{y} |x)$ , is transformed into a classification problem of estimating  $p(k|x)$  in a discrete space. Obviously, the size of the bin,  $|\mathcal{I}_j|$ , affects the fidelity of the approximation. The effects of the bin size are presented in Section 3.1. There is a similarity between the discretization and the idea of Lin et al. (2007). However, it should be noted that the same discretization technique, often called "finite volume method", has been widely used in the numerical simulations of partial differential equations for a long time.

The discretization naturally leads to the conventional cross-entropy (CE) minimization. Suppose we have a data set,  $D_R = \{(\hat{y}_i,x_i);\hat{y}_i\in \mathbb{R},x_i\in \mathbb{R},$  and  $i = 1,\dots ,N\}$ . We can define a mapping  $\mathcal{C}:\mathbb{R}\to \mathbb{N}_{+}$  such that  $\mathcal{C}(\hat{y}) = k$ , if  $y\in \mathcal{I}_k$ . Then,  $D_{R}$  can be easily converted to a new data set for target labels,  $D_{C} = \{(c_{i},\hat{y}_{i},x_{i});c_{i}\in \mathbb{N}_{+},\hat{y}_{i}\in \mathbb{R},x_{i}\in \mathbb{R},$  and  $i = 1,\ldots ,N\}$ , where  $c_{i} = \mathcal{C}(\hat{y}_{i})$ .  $D_{C}$  provides a training data set for the following CE minimization problem,

$$
C E = - \sum_ {n = 1} ^ {N} \sum_ {k = 1} ^ {K} \delta_ {c _ {n} k} \log P _ {k} ^ {n} = - \sum_ {n = 1} ^ {N} \log P _ {c _ {n}} ^ {n}. \tag {13}
$$

Note, however, that the CE minimization does not explicitly guarantee the smoothness of the estimated distribution. Since CE loss function depends only on  $P_{i}$  of a correct label,  $\delta_{c_n k}$ , as a result, in the optimization problem every element  $P_{i}$ , except for the one corresponding to the correct label,  $P_{c_n}$ , is penalized in the same way, which is natural in the conventional classification tasks where a geometric proximity between the classes is not relevant. In the present study, however, the softmax layer, or class probability, is used as a discrete approximation to a smooth function. Hence, it is expected that  $P_{c_n}$  and  $P_{c_n \pm 1}$  (i.e., the nearby classes) should be close to each other. To address this issue, in the following Sections 2.2.1 and 2.2.2, we propose two types of regularization to impose the class proximity structure in the CE loss.

# 2.2.1 EXPLICIT REGULARIZATION OF CROSS-ENTROPY LOSS

To explicitly impose the smoothness between the classes, we propose to use a regularized cross-entropy (RCE) minimization, defined by the following loss function

$$
\mathrm {R C E} = \sum_ {n = 1} ^ {N} \left\{\sum_ {k = 1} ^ {K} - \delta_ {c _ {n} k} \log P _ {k} ^ {n} + \lambda \left(\boldsymbol {L} \boldsymbol {P} ^ {n}\right) ^ {T} \boldsymbol {L} \boldsymbol {P} ^ {n} \right\}, \tag {14}
$$

where  $\lambda$  is a penalty parameter and the Laplacian matrix  $L\in \mathbb{R}^{K - 2,K}$  is

$$
\boldsymbol {L} = \left[ \begin{array}{c c c c c c} 1 & - 2 & 1 & 0 & \dots & 0 \\ 0 & 1 & - 2 & 1 & \dots & 0 \\ \dots & \dots & \dots & \dots & \dots & \dots \\ 0 & \dots & 0 & 1 & - 2 & 1 \end{array} \right]. \tag {15}
$$

RCE is analogous to the penalized maximum likelihood solution for density estimation (Silverman, 1986). Assuming a uniform bin size,  $|\mathcal{I}_0| = \dots = |\mathcal{I}_K| = \delta y$ , the Laplacian of a distribution can be approximated by a Taylor expansion  $p''(\hat{y} |x)|_{y = \alpha_{i - 1 / 2}}\simeq (P_{i - 1} - 2P_i + P_{i + 1}) / \delta y^2$ , where  $\alpha_{i - 1 / 2} = 0.5(\alpha_{i - 1} + \alpha_i)$ . Then, it is clear that

$$
\left(\boldsymbol {L} \boldsymbol {P} ^ {n}\right) ^ {T} \boldsymbol {L} \boldsymbol {P} ^ {n} \sim \int \left[ p ^ {\prime \prime} (\hat {y} | x) \right] ^ {2} d y. \tag {16}
$$

In other words, RCE aims to smooth out the distribution by penalizing local minima or maxima.

# 2.2.2 IMPLICIT REGULARIZATION OF CROSS-ENTROPY LOSS

Alternative to adding an explicit regularization to CE, the smoothness can be achieved by enforcing a spatial correlation in the network output. Here, we use an one-dimensional convolution layer to enforce smoothness. Let  $\widetilde{\pmb{o}}\in \mathbb{R}^{K}$  be the input to the softmax layer in DE-DNN. We can add a convolution layer,  $\pmb {o}\in \mathbb{R}^{K}$ , on top of  $\widetilde{o}$ , such that

$$
o _ {i} = \sum_ {j = 1} ^ {K} \frac {1}{h} \exp \left[ - \frac {1}{2} \left(\frac {i - j}{h}\right) ^ {2} \right] \widetilde {o} _ {j}, \text {f o r} i = 1, \dots , K, \tag {17}
$$

where the penalty parameter  $h$  determines the smoothness of the estimated distribution. Using (17), the LSTM model can now be trained by the standard CE. The implicit regularization, here we call convolution CE (CCE), is analogous to a kernel density estimation.

# 2.3 COMPUTING TIME EVOLUTION OF PROBABILITY DISTRIBUTION

In the next-step prediction of DE-RNN, the inputs are  $(\hat{y}_t,\pmb {u}_t)$ , and the output is the probability distribution,

$$
\pmb {P} _ {t + 1} = \Psi_ {d} (\pmb {s} _ {t}) = \Psi_ {d} \circ \Psi_ {e} (\hat {y} _ {t}, \pmb {u} _ {t}, \pmb {s} _ {t - 1}).
$$

Note that  $D_C$  is used only in the training stage. Then, the moments of the predictive distribution can be easily evaluated, e.g.,

$$
E \left[ \hat {y} _ {t + 1} \mid \widehat {\mathbf {Y}} _ {0: t}, U _ {0: t} \right] = \boldsymbol {\alpha} _ {1 / 2} ^ {T} \boldsymbol {P} _ {t + 1}, \operatorname {V a r} \left[ \hat {y} _ {t + 1} \mid \widehat {\mathbf {Y}} _ {0: t}, U _ {0: t} \right] = \left(\boldsymbol {\alpha} _ {1 / 2} ^ {2}\right) ^ {T} \boldsymbol {P} _ {t + 1} - E \left[ \hat {y} _ {t + 1} \mid \widehat {\mathbf {Y}} _ {0: t}, U _ {0: t} \right] ^ {2}, \tag {18}
$$

where  $\alpha_{1/2} = (\alpha_{1/2}, \alpha_{1+1/2}, \dots, \alpha_{K-1/2})^T$ ,  $\alpha_{1/2}^2 = \alpha_{1/2} \odot \alpha_{1/2}$ , and  $\alpha_{i-1/2} = 0.5(\alpha_{i-1} + \alpha_i)$ .

Next, we consider a multiple-step forecast, which corresponds to computing a temporal evolution of the probability distribution, i.e.,  $p(\hat{y}_{t + n}|\widehat{\mathbf{Y}}_{0:t},\mathbf{U}_{0:t + n - 1})$  for  $n > 1$ .

Applying the results of Section 2.2, once the distribution of  $\hat{y}_{t + 1}$  in (10) is computed, the distribution of  $\hat{y}_{t + 2}$  can be similarly obtained as  $p(\hat{y}_{t + 2}|s_{t + 1})$ . Observe that  $s_{t + 1}$  is computed from a deterministic function of  $s_t$ ,  $u_{t + 1}$ , and  $\hat{y}_{t + 1}$ , i.e.,

$$
\pmb {s} _ {t + 1} = \Psi_ {e} (\hat {y} _ {t + 1}, \pmb {u} _ {t + 1}, \pmb {s} _ {t}).
$$

Here,  $\pmb{u}_{t+1}$  and  $\pmb{s}_t$  are already known, while  $\hat{y}_{t+1}$  is a random variable, whose distribution  $p(\hat{y}_{t+1}|s_t)$  is computed from the deterministic function  $\Psi_d(s_t)$ . Then,  $s_{t+1}$  is also a random variable. The distribution,  $p(\pmb{s}_{t+1}|s_t,\pmb{u}_{t+1})$ , can be obtained by applying a change of variables on  $p(\hat{y}_{t+1}|s_t)$  with a nonlinear mapping  $\Psi_e$ . Repeating this process, the multiple-step-ahead predictive distribution can therefore be computed as

$$
p \left(\hat {y} _ {t + n} \mid \widehat {\mathbf {Y}} _ {0: t}, \boldsymbol {U} _ {0: t + n - 1}\right) = \int \dots \int p \left(\hat {y} _ {t + n} \mid \boldsymbol {s} _ {t + n - 1}\right) \prod_ {i = 1} ^ {n - 1} p \left(\boldsymbol {s} _ {t + i} \mid \boldsymbol {s} _ {t + i - 1}, \boldsymbol {u} _ {t + i}\right) d \boldsymbol {s} _ {t + i}. \tag {19}
$$

Since the high dimensional integration in (19) is intractable, we propose to approximate it by a sequential Monte Carlo method. The Monte Carlo procedure is outlined in Algorithm 1.

# Algorithm 1 Sequential Monte Carlo method for LSTM multi-step-ahead prediction

Input:  $\widehat{Y}_{0:t}$ ,  $U_{0:t}$ , number of Monte Carlo samples,  $N_s$ , and forecast horizon  $n$

Output:  $p(\hat{y}_{t + n}|\widehat{\mathbf{Y}}_{0:t},\mathbf{U}_{0:t + n - 1})$  (density estimation from  $\hat{\mathbf{y}}_{t + n}$ )

Initialization: Set LSTM states to  $s_0 = h_0 = 0$

Perform a sequential update of LSTM up to time  $t$  from the noisy observations  $(\widehat{Y}_{0:t})$ .

$$
\boldsymbol {s} _ {i} = \Psi_ {e} (\hat {y} _ {i}, \boldsymbol {u} _ {i}, \boldsymbol {s} _ {i - 1}) \text {f o r} i = 1, \dots , t.
$$

Make  $N_{s}$  replicas of the internal state,  $s_t^1 = \dots = s_t^{N_s} = s_t$

repeat

Compute the predictive distribution of  $\hat{y}_{t + 1}^{i}$  for each sample

$$
\boldsymbol {P} _ {t + 1} ^ {i} = \Psi_ {d} (\boldsymbol {s} _ {t} ^ {i}), \text {f o r} i = 1, \dots , N _ {s}.
$$

Sample the target variable at  $t + 1$ ,  $\hat{y}_{t + 1}^{i}$ , from the distribution

1. Sample the class label from the discrete distribution:  $c^i \sim P_{t+1}^i$  
2. Sample  $\hat{y}_{t + 1}^i$  in  $\mathcal{I}_{c^i}\colon \hat{y}_{t + 1}^i\sim \mathcal{U}(\alpha_{c^i -1},\alpha_{c^i})$

Update the internal state of LSTM

$$
\boldsymbol {s} _ {t + 1} ^ {i} = \Psi_ {e} (\hat {y} _ {t + 1} ^ {i}, \boldsymbol {u} _ {t + 1}, \boldsymbol {s} _ {t} ^ {i}).
$$

until (all  $\hat{\pmb{y}}_{t + n}$  are sampled)

![](images/edec1a9d2fbeb87ce8cb18d101368832c2ea962e226a152745fb6c11d4e437d4.jpg)  
Figure 1: NRMSE of the next-step prediction of the CIR process by RCE (a,b) and CCE (c,d). The bin size is (a,c)  $\delta y = 0.08$  and (b,d) 0.04. The hollow circles  $(\circ)$  denote NRMSE in the expectation  $(e_{\mu})$  and the solid circles  $(\bullet)$  are the standard deviation  $(e_{\sigma})$ .

![](images/01aafc2e65f800b7fe5e9310f43c4d32d9ae9ca3e1a9bc69105632e6b4b69d93.jpg)

![](images/cfab27ff25b9f769b7aacb2275b460492524ee2fb1683a60515df421951a45a5.jpg)

![](images/2f9fcc9bc886c6de92f634df319cd98f7e5aaeef4bc32a3d0b3808f7a0cff999.jpg)

# 3 EXPERIMENTS

In this section, DE-RNN is tested against two synthetic and two real data sets. The LSTM architecture used in all of the numerical experiments is identical. Two feed-forward networks are used before and after the LSTM;

$$
\boldsymbol {z} _ {t} = \mathcal {L} \left(\varphi_ {T} \circ \mathcal {L} \left(y _ {t}, \boldsymbol {u} _ {t}\right) + \boldsymbol {h} _ {t - 1}\right), \quad \boldsymbol {P} _ {t + 1} = \varphi_ {S M} \circ \mathcal {L} \left(\varphi_ {T} \circ \mathcal {L} \left(\varphi_ {S P} \circ \mathcal {L} \left(\boldsymbol {h} _ {t}\right)\right)\right), \tag {20}
$$

in which  $\varphi_{SP}$  and  $\varphi_{SM}$  denote the softplus and softmax functions, respectively. The size of the bins is kept uniform, i.e.,  $|\mathcal{I}_1| = \dots = |\mathcal{I}_K| = \delta y$ . The LSTM is trained by using ADAM (Kingma & Ba, 2015) with a minibath size of 20 and a learning rate of  $\eta = 10^{-3}$ .

# 3.1 COX-INGERSOLL-ROSS PROCESS

First, we consider a modified Cox-Ingersoll-Ross (CIR) process, which is represented by the following stochastic differential equation,

$$
d y (t) = - 0. 5 y (t) d t + \sqrt {0 . 5 + | y (t) |} d W, \tag {21}
$$

in which  $W$  is the Weiner process. The original CIR process is used to model the valuation of interest rate derivatives in finance. Equation (21) is solved by the forward Euler method with the time step size  $\delta t = 0.1$ . The simulation is performed for  $T = (0,160000]\delta t$  to generate the training data and  $T = (160000,162000]\delta t$  is used for the testing. Note that the noise component of CIR is multiplicative, which depends on  $y(t)$ .

The experiments are performed for two different bin sizes,  $dy = 0.08$  and 0.04. The DE-RNN has 64 LSTM cells. Figure 1 shows the errors in the expectation and the standard deviation with respect to the analytical solution;

$$
E _ {y \sim p _ {T}} \left[ y _ {t + 1} \mid y _ {t} \right] = y _ {t} \exp (- 0. 5 \delta t), \quad s d _ {y \sim p _ {T}} \left[ y _ {t + 1} \mid y _ {t} \right] = \sqrt {\left(0 . 5 + \left| y _ {t} \right|\right) \delta_ {t}}. \tag {22}
$$

Here,  $p_T$  denotes the true distribution of the CIR process. The normalized root mean-square errors (NRMSE) are defined as

$$
e _ {\mu} = \frac {\left\langle \left(E _ {y \sim p _ {L}} \left[ y _ {t + 1} \mid y _ {t} \right] - E _ {y \sim p _ {T}} \left[ y _ {t + 1} \mid y _ {t} \right]\right) ^ {2} \right\rangle^ {1 / 2}}{\left\langle \left(y _ {t} - E _ {y \sim p _ {T}} \left[ y _ {t + 1} \mid y _ {t} \right]\right) ^ {2} \right\rangle^ {1 / 2}}, \tag {23}
$$

$$
e _ {\sigma} = \frac {\left\langle \left(s d _ {y \sim p _ {L}} \left[ y _ {t + 1} \mid y _ {t} \right] - s d _ {y \sim p _ {T}} \left[ y _ {t + 1} \mid y _ {t} \right]\right) ^ {2} \right\rangle^ {1 / 2}}{s d [ y ]}, \tag {24}
$$

in which  $\langle \cdot \rangle$  denotes an average over the testing data,  $p_L$  is the distribution from the LSTM, and  $sd[y]$  denotes the standard deviation of the data. The error in the expectation is normalized against a zeroth-order prediction, which assumes  $y_{t + 1} = y_t$ .

In Figure 1, it is clearly shown that the prediction results are improved when a regularization is used to impose a smoothness condition. Comparing Figures 1 (a) and (b), for RCE,  $e_{\mu}$  and  $e_{\sigma}$  become smaller when a smaller  $\delta y$  is used. As expected,  $e_{\sigma}$  increases when  $\lambda$  is large. But, for the smaller bin size,  $\delta y = 0.04$ , both  $e_{\mu}$  and  $e_{\sigma}$  are not so sensitive to  $\lambda$ . Similar to RCE,  $e_{\mu}$  and  $e_{\sigma}$  for CCE decrease at first as the penalty parameter  $h$  increases. However, in general, RCE provides a better prediction compared to CCE.

Table 1: NRMSE of the mean and standard deviation of the next-step prediction. The DE-RNN results are compared with the first-order autoregressive model (AR) and Kalman filter (KF).  

<table><tr><td></td><td>CE</td><td>RCE</td><td>CCE</td><td>AR(1)</td><td>KF</td></tr><tr><td>eμ</td><td>0.238</td><td>0.0549</td><td>0.149</td><td>0.029</td><td>0.029</td></tr><tr><td>eσ</td><td>0.066</td><td>0.017</td><td>0.038</td><td>0.228</td><td>0.228</td></tr></table>

![](images/9dfca19c98b628c595add81971ec9b0f7470ca147c6db30f1e237273a1fd64e5.jpg)  
Figure 2: 200-step forecast of (a) expectation and (b) standard deviation of the CIR process. The circles denote the solution of Eqn (21) from a Monte Carlo method with  $10^{7}$  samples.

![](images/e2cdd1da4185576ab646559f4d4b4547dc6d80d45dacc51122bba69e25adaec6.jpg)

NRMEs are listed in Table 1. For a comparison, the predictions by AR(1) and KF are shown. The CIR process is essentially a first-order autoregressive process. So, it is not surprising to see that AR(1) and KF, which are designed for the first-order AR process, outperforms DE-RNN for the prediction of the expectation. However,  $e_{\sigma}$  of AR(1) and KF are much larger than that of DE-RNN, because those models assume an additive noise. Note that  $e_{\sigma}$  of RCE and CCE are less than  $4\%$ , suggesting that DE-RNN can model the complex noise process very well.

In Figure 2, a 200-step forecast by DE-RNN is compared with a Monte-Carlo solution of equation (21). DE-RNN is trained with  $\delta y = 0.04$  and  $\lambda = 200$ . For the DE-RNN forecast, the testing data is supplied to DE-RNN for the first 100 time steps, i.e., for  $t = -10$  to  $t = 0$ , and the SMC multiple-step forecast is performed for the next 200 time steps with 20,000 samples. It is shown that the multiple-step forecast by DE-RNN agrees very well with the MC solution of the CIR process. Note that, in Figure 2 (b), the noise process, as reflected in  $sd[y_t]$ , is a function of  $y_t$ , and hence the multi-step forecast of the noise increases rapidly first and then decreases before reaching a plateau. The SMC forecast can accurately capture the behavior. Such kind of behavior can not be represented if a simple additive noise is assumed.

# 3.2 MACKKEY-GLASS TIME SERIES

For the next test, we applied DE-RNN for a time series generated from the Mackey-Galss equation (Mackey & Glass, 1977);

$$
\frac {d y}{d t} = \frac {\alpha y (t - \tau)}{1 + y ^ {\beta} (t - \tau)} - \gamma y (t). \tag {25}
$$

We use the parameters adopted from Gers (2001),  $\alpha = 0.2$ ,  $\beta = 10$ ,  $\gamma = 0.1$ , and  $\tau = 17$ .

The Mackey-Glass equation is solved by using a third-order Adams-Bashforth method with a time step size of 0.02. The time series is generated by down-sampling, such that the time interval between consecutive data is  $\delta t = 1$ . A noisy observation is made by adding a white noise;

$$
\hat {y} _ {t} = y _ {t} + \epsilon_ {t}.
$$

$\epsilon_{t}$  is a zero-mean Gaussian random variable with the noise level  $sd[\epsilon_t] = 0.3sd[y]$ . A time series of the length  $1.6 \times 10^{5}\delta t$  is generated for the model training and another time series of length  $2 \times 10^{3}\delta t$  is made for the validation. DE-RNN is trained for  $\delta y = 0.04sd[y]$  and consists of 128 LSTM cells.

Figure 3 (a) shows the noisy observation and the expectation of the next-step prediction,  $E[\hat{y}_{t + 1}|\hat{y}_t]$ , in a phase space. It is shown that DE-RNN can filter out the noise and reconstruct the original dynamics accurately. Even though the noisy data are used as an input,  $E[\hat{y}_{t + 1}|\hat{y}_t]$  accurately represents the original attractor of the chaotic system, indicating a strong de-noising capability of the LSTM.

![](images/1726ac6fbb4e2aa34ed12003489b9ca5026597d9238ebf2a84bab45b61631c11.jpg)  
Figure 3: (a) Next-step prediction  $(\bullet)$  and the noisy observation  $(\odot)$  for the Mackey-Glass equation. The solid line denotes the ground truth,  $y(t)$ . (b) The next-step probability distribution,  $p(\hat{y}_{t+1}|\hat{y}_t)$ , from the standard CE  $(\odot)$  and CCE  $(\bullet)$  with  $h = 5$ .

![](images/e62e6ef0d200f95d7c14988e48ee2399d4f811890b1666a7b4e10f496094c2ad.jpg)

Table 2: NRMSEs of the Mackey-Galss time series. DE-RNN results are compared with autoregressive integrated moving average (ARIMA) and Kalman filter (KF).  

<table><tr><td rowspan="2"></td><td colspan="4">RCE</td><td colspan="2">CCE</td><td rowspan="2">ARIMA</td><td rowspan="2">KF</td></tr><tr><td>λ = 0</td><td>λ = 50</td><td>λ = 100</td><td>λ = 200</td><td>h = 5</td><td>h = 10</td></tr><tr><td>eμ</td><td>0.143</td><td>0.141</td><td>0.143</td><td>0.142</td><td>0.133</td><td>0.143</td><td>0.668</td><td>0.916</td></tr><tr><td>eσ</td><td>0.032</td><td>0.023</td><td>0.027</td><td>0.038</td><td>0.013</td><td>0.020</td><td>0.191</td><td>0.351</td></tr></table>

The estimated probability distribution is shown in Figure 3 (b). Without a regularization, the standard CE results in a noisy distribution, while the distribution from CCE shows a smooth Gaussian shape.

The prediction errors are shown in table 2. NRMSEs are defined as,

$$
e _ {\mu} = \frac {\left\langle \left(E \left[ \hat {y} _ {t + 1} \mid \hat {y} _ {t} \right] - y _ {t + 1}\right) ^ {2} \right\rangle^ {1 / 2}}{\left\langle \left(\hat {y} _ {t} - y _ {t + 1}\right) ^ {2} \right\rangle^ {1 / 2}}, \quad e _ {\sigma} = \frac {s d \left[ \hat {y} _ {t + 1} \mid \hat {y} _ {t} \right]}{s d \left[ \epsilon_ {t} \right]} - 1, \tag {26}
$$

NRMSEs are computed with respect to the ground truth. Again,  $e_{\mu}$  compares the prediction error to the zeroth-order prediction. In this example, the errors are not so sensitive to the regularization parameters. The best result is achieved by CCE. DE-RNN can make a very good estimation of the noise. The error in the noise component,  $e_{\sigma}$ , is only  $2\% \sim 5\%$ . Unlike the CIR process, NRMSEs from KF and ARIMA are much larger than those of DE-RNN. Because the underlying process is a delay-time nonlinear dynamical system, those linear models can not accurately approximate the complex dynamics.

A multiple-step forecast of the Mackey-Glass time series is shown in Figure 4. In the validation time series, the observations in  $t \in [1,100] \delta t$  are supplied to the DE-RNN to develop the internal state, and a 500-step forecast is made for  $t \in [101,600] \delta t$ . In Figure 4 (a), it is shown that a multiple-step forecast by a standard regression LSTM approximates  $y(t)$  very well initially, e.g., for  $t < 80 \delta t$ , but

![](images/e598730e598abab21be6fbaadd953e2e08acb7095c57eaddaaad71f5b7350725.jpg)  
Figure 4: (a) 500-step forecast by a regression LSTM  $(\circ)$  and the ground truth  $(-)$ . (b) The color contours denote a 500-step forecast of the probability distribution,  $p(\hat{y}_{n + s}|\hat{y}_s)$ , and the dashed lines are  $95\%$  -CI. The ground truth is shown as the solid line  $(-)$ .

![](images/4ff80d63fe807db8fa8da9f7b0c40e853cdbeff8ffbd33befd852ee2bac32185.jpg)

![](images/da9674c6c42322e45a779971a0ccf99642f7c4bbe6b49542db7d3bd654aab7de.jpg)  
Figure 5: (a) Mauna Loa  $\mathrm{CO}_{2}$  observation. The vertical line denotes the boundary of the training and testing data. (b) 17-year forecast of the  $\mathrm{CO}_{2}$  concentration: Apr-01-2000 Sep-23-2017. The solid and dashed lines denote the expectation and  $95\%$  -CI, respectively. The observation is shown as the solid circles  $(\bullet)$  and a regression LSTM is the hollow circles  $(\circ)$ . The time unit is a week.

![](images/918199ba2742fcf7b028d48d3ed2ceb2475df9b08a6dde6a3bf625fe41774286.jpg)

eventually diverges for larger  $t$ . Because of the Mackey-Glass time series is chaotic, theoretically it is impossible to make a long time forecast. But, in the DE-RNN forecast,  $y(t)$  is bounded by the 95%-confidence interval (CI) even for the 500-step forecast. Note that the uncertainty, denoted by 95%-CI grows only at a very mild rate in time. In fact, it is observed that CI is not a monotonic function of time. In DE-RNN, the 95%-CI may grow or decrease following the dynamics of the system, while for the conventional time series models, such as ARIMA and KF, the uncertainty is a non-decreasing function of time.

# 3.3 MAUNA LOA  $\mathrm{CO}_{2}$  OBSERVATION

In this experiments, DE-RNN is tested against the atmospheric  $\mathrm{CO}_{2}$  observation at Mauna Loa Observatory, Hawaii (Keeling et al., 2001). The  $\mathrm{CO}_{2}$  data set consists of weekly-average atmospheric  $\mathrm{CO}_{2}$  concentration from Mar-29-1958 to Sep-23-2017 (Figure 5 a). The data from Mar-29-1958 to Apr-01-2000 is used to train DE-RNN and a 17-year forecast is made from Apr-01-2000 to Sep-23-2017. This  $\mathrm{CO}_{2}$  data has been used in previous studies (Gal & Ghahramani, 2016; Rasmussen & Williams, 2006). In DE-RNN, 64 LSTM cells and  $\delta y = 0.1sd[dy_{t}]$ , in which  $dy_{t} = y_{t+1} - y_{t}$ , are used.

The 17-year DE-RNN forecast, with 1,000 MC samples, is shown in Figure 5 (b). DE-RNN well represents the growing trend and the oscillatory pattern of the  $\mathrm{CO}_{2}$  data. The  $\mathrm{CO}_{2}$  data is nonstationary, where the rate of increase of  $\mathrm{CO}_{2}$  is an increasing function of time. Since DE-RNN is trained against the history data, where the rate of  $\mathrm{CO}_{2}$  increase is smaller than the current, it is expected that the forecast will underestimate the future  $\mathrm{CO}_{2}$ .  $E[\hat{y}_{n + s}|\hat{y}_s]$  agrees very well with the observation for the first 200 weeks, but eventually underestimates  $\mathrm{CO}_{2}$  concentration. It is interesting to observe that the upper bound of the  $95\%$  -CI grows more rapidly than the expectation and provides a good approximation of the observation. For a comparison, the forecast by a regression LSTM is also shown. Similar to the chaotic Mackey-Glass time series, the regression LSTM makes a good prediction for a short time, e.g.,  $t < 100$  weeks, but eventually diverges from the observation. Note that the lower bound of  $95\%$  -CI encompasses the regression LSTM.

# 3.4 CPU TEMPERATURE FORECAST

In the last experiment, IBM Power System S822LC and NAS Parallel Benchmark (NPB) are used to generate the temperature trace. Figure 6 (a) shows the temperature of a CPU. The temperature sensor generates a discrete data, which has a resolution of  $1^{\circ}\mathrm{C}$ . The CPU temperature is controlled by three major parameters; CPU frequency, CPU utilization, and cooling fan speed. In this experiment, we have randomized the CPU frequencies and job arrival time to mimic the real workload behavior, while the fan speed is fixed to 3300RPM. The time step size is  $\delta t = 2$  seconds. Accurate forecast of CPU temperature for a future workload scenario is essential in developing an energy-efficient control strategy for the thermal management of a cloud system.

Figure 6 (c) and (d) show multiple-step forecasts of the CPU temperature by RCE and a regression LSTM, respectively. The bin size is  $\delta y = 0.18^{\circ}\mathrm{C}$ , which is smaller than the sensor resolution. In the forecast, the future control parameters are given to DE-RNN. In other words, DE-RNN

![](images/bd7307bfe307bb7e6faf1d124c2199024ef1e8d10a0304c072df7cd187f3b80b.jpg)

![](images/42a8c56fde831dc237871019dabcb7c8e7241fe4de85fe025a8b4ee248c786ea.jpg)

![](images/564a2cd127c67665a08fdeec30a1f9f70899a930dd75ee7d418fccff18b37fc3.jpg)  
Figure 6: (a) Temperature of a CPU in  $^\circ \mathrm{C}$ . (b) Control parameters; CPU utilization (black) and Clock speed (blue). Multiple-step forecasts by (c) RCE and (d) regression LSTM. The solid and dashed lines in (c) denote  $E[\hat{y}_{n + s}|\hat{y}_s]$  and  $95\%$ -CI, respectively. The circles are the observations.

![](images/ea572ecf3754ddd51b5ac135a760ed56b44b36663cf0dee060c5c277ec6e1f7a.jpg)

Table 3:  ${l}_{\infty }$  error for RCE,CCE and regression LSTM in  ${}^{ \circ  }\mathrm{C}$  .  

<table><tr><td>|E[hat{y}_{n+s}|hat{y}_s] - \hat{y}_{n+s}||∞</td><td>RCE 4.61</td><td>CCE 2.70</td><td>LSTM 9.48</td></tr></table>

predicts the probability distribution of future temperature with respect to a control scenario, i.e.,  $p(\hat{y}_{t + n}|\widehat{\mathbf{Y}}_{0:t},\mathbf{U}_{0:t + n - 1})$ . The forecast is made by using 5,000 Monte Carlo samples. Here, 1800-step forecast is made,  $t = 0\sim 3,600$  sec. and only the results in  $t\in (50,1800)$  sec. is shown. While the regression LSTM makes a very noisy prediction near the local peak temperature at  $t\simeq 500$ , RCE provides a much more stable forecast. Table 3 shows the  $l_{\infty}$ -errors, i.e., maximum absolute difference. The maximum error is observed near the peak temperature at  $t\simeq 500$ . ARIMA and KF are also tested for the multiple-step forecast, but the results are not shown because their performance is much worse than the LSTMs. The changes in the temperature are associated with step changes of some control parameters. Such abrupt transitions seem to cause the large oscillation in the regression LSTM prediction. But, for RCE or CCE, the prediction is made from an ensemble of Monte Carlo samples, which makes it more robust to such abrupt changes. Also, note that for  $t < 200$  sec., RCE prediction  $(\simeq 53.4^{\circ}\mathrm{C})$  is in between the two discrete integer levels,  $53^{\circ}\mathrm{C}$  and  $54^{\circ}\mathrm{C}$ , which correctly reflects the uncertainty in the measurement, while the regression LSTM  $(\simeq 53.1^{\circ}\mathrm{C})$  more closely follows one of the two levels.

# 4 CONCLUDING REMARKS

We present DE-RNN to compute the time evolution of a probability distribution for complex time series data. DE-RNN employs LSTM to learn multiscale, nonlinear dynamics from the noisy observations, which is supplemented by a softmax layer to approximate a probability density function. To assign probability to the softmax output, we use a mapping from  $\mathbb{R}$  to  $\mathbb{N}_{+}$ , which leads to a cross-entropy minimization problem. To impose a geometric structure in the distribution, two regularization strategies are proposed. The regularized cross-entropy method is analogous to the penalized maximum likelihood estimate for the density estimation, while the convolution cross-entropy method is motivated by the kernel density estimation. The proposed algorithm is validated against two synthetic data set, for which we can compare with the analytical solutions, and two real data sets. In this study, for simplicity, the problem is formulated for a univariate time series. But, it is straightforward to extend the methodology to a multivariate time series.

# REFERENCES

E. Archer, M. Park, L. Buesing, J. Cunningham, and L. Paninski. Black box variational inference for state space models. arXiv preprint arXiv:1511.07367, 2015.  
J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.  
J. Chung, K. Kastner, L. Dinh, K. Goel, A. Courville C, and Y. Bengio. A recurrent latent variable model for sequential data. In Advances in Neural Information Processing Systems, pp. 2980-2988, 2015.  
S. Dasgupta and T. Osogami. Nonlinear dynamic boltzmann machines for time-series prediction. In AAAI Conference on Artificial Intelligence, 2017.  
Y. Gal and Z. Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In Proceedings of the 33rd International Conference on International Conference on Machine Learning, pp. 1050-1059, 2016.  
F.A.Gers.Ph.D Thesis.EPFL,2001.  
F. A. Gers, J. Schmidhuber, and F. Cummins. Learning to forget: Continual prediction with LSTM. Neural Comput., 12:2451 - 2471, 2000.  
A. C. Harvey. Forecasting, structural time series models and the Kalman filter. Cambridge university press, 1990.  
S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 9:1735 - 1780, 1997.  
D. Hsu. Time series forecasting based on augmented long short-term memory. arXiv preprint arXiv:1707.00666, 2017.  
C. D. Keeling, S. C. Piper, R. B. Bacastow, M. Wahlen, T. P. Whorf, M. Heimann, and H. A. Meijer. Exchanges of atmospheric  $\mathrm{CO}_{2}$  and  $^{13}\mathrm{CO}_{2}$  with the terrestrial biosphere and oceans from 1978 to 2000. I. Global aspects. In SIO Reference Series, pp. No. 01-06. Scripps Institution of Oceanography, 2001.  
D. P. Kingma and J. L. Ba. ADAM: A method for stochastic optimization. In 3rd International Conference on Learning Representation, San Diego, CA, USA, 2015.  
R. Krishnan, U. Shalit, and D. Sontag. Structured inference networks for nonlinear state space models. In AAAI Conference on Artificial Intelligence, 2017.  
H. Lasi, P. Fettke, H. Kemper, T. Feldand, and M. Hoffmann. Industry 4.0. Business & Information Systems Engineering, 6(4):239-242, 2014.  
J. Lin, E. Keogh, L. Wei, and S. Lonardi. Experiencing SAX: a novel symbolic representation of time series. Data Mining and knowledge discovery, 15(2):107-144, 2007.  
H. Lütkepohl. New introduction to multiple time series analysis. Springer Science & Business Media, 2005.  
M. Mackey and L. Glass. Oscillation and chaos in physiological control systems. Science, 197: 287-289, 1977.  
Y. Qin, D. Song, H. Chen, W. Cheng, G. Jiang, and G. W. Cottrell. A dual-stage attention-based recurrent neural network for time series prediction. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17, pp. 2627-2633, 2017.  
C. E. Rasmussen and K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.  
B. W. Silverman. Density estimation for statistics and data analysis. Chapman & Hall, 1986.  
W.-X. Wang, Y.-C. Lai, and C. Grebogi. Data based identification and prediction of nonlinear and complex dynamical systems. Phys. Reports, 644:1-76, 2016.

Z. Zhang and G. E. Karniadakis. Numerical methods for stochastic partial differential equations with white noise. Springer, 2017.  
L. Zhu and N. Laptev. Deep and confident prediction for time series at Uber. arXiv preprint arXiv:1709.01907, 2017.
