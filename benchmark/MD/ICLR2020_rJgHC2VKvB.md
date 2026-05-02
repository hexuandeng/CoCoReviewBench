# RECURRENT NEURAL NETWORKS ARE UNIVERSAL FILTERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recurrent neural networks (RNN) are powerful time series modeling tools in machine learning. It has been successfully applied in a variety of fields such as natural language processing (Mikolov et al. (2010), Graves et al. (2013), Du et al. (2015)), control (Fei & Lu (2017)) and traffic forecasting (Ma et al. (2015)), etc. In those application scenarios, RNN can be viewed as implicitly modelling a stochastic dynamic system. Another type of popular neural network, deep (feed-forward) neural network has also been successfully applied in different engineering disciplines, whose approximation capability has been well characterized by universal approximation theorem (Hornik et al. (1989), Park & Sandberg (1991), Lu et al. (2017)). However, the underlying approximation capability of RNN has not been fully understood in a quantitative way. In our paper, we consider a stochastic dynamic system with noisy observations and analyze the approximation capability of RNN in synthesizing the optimal state estimator, namely optimal filter. We unify the recurrent neural network into Bayesian filtering framework and show that recurrent neural network is a universal approximator of optimal finite dimensional filters under some mild conditions. That is to say, for any stochastic dynamic systems with noisy sequential observations that satisfy some mild conditions, we show that (informal)

$$
\forall \epsilon > 0, \exists \operatorname {R N N - b a s e d f i l t e r}, \text {s . t .} \lim _ {k \rightarrow \infty} \sup  _ {\epsilon <   0} \left\| \hat {x} _ {k | k} - \mathbb {E} [ x _ {k} | Y _ {k} ] \right\| <   \epsilon ,
$$

where  $\hat{x}_{k|k}$  is RNN-based filter's estimate of state  $x_{k}$  at step  $k$  conditioned on the observation history and  $\mathbb{E}[x_k|Y_k]$  is the conditional mean of  $x_{k}$ , known as the optimal estimate of the state in minimum mean square error sense. As an interesting special case, the widely used Kalman filter (KF) can be synthesized by RNN.

# 1 INTRODUCTION

Recurrent neural network (RNN) is a certain type of neural networks characterized by hidden variables that memorize the history of input sequences, and it has been successfully applied and brought amazing results in many different disciplines including computer vision, natural language processing and optimal control, etc. (Mikolov et al. (2010), Graves et al. (2013), Du et al. (2015), Fei & Lu (2017), Ma et al. (2015)). Its huge empirical success in different engineering disciplines is grounded on the expressive power of RNN. However, how to understand the expressive power of RNN in a quantitative way is not fully understood. Even what RNN expresses is not totally clear. Another type of neural network, deep feed forward neural network has been well characterized as a universal function approximator (Hornik et al. (1989), Park & Sandberg (1991)). However, a similar way to characterize the expressive power of RNN is not obvious.

DNN is a mapping from a finite dimensional Euclidean space to another finite dimensional Euclidean space, that is to say it can be regarded as a vector-valued multivariate function. However, RNN is a mapping from a sequence space to another sequence space and the current output depends on both current input and the whole observation history. The input sequence, in principle, can be infinite. The function of RNN is capturing the relationship between input process and output process. For example, in machine translation, the input process (or the observation process) is sentence in one language and the output process (or the state process) is sentence in another language. And in many other RNN's application scenarios such as traffic forecast and optimal control, the input is a noisy

observation or measurement sequence and the output is an estimate sequence of a certain quantity, e.g., the traffic speed. We observe that the function of RNN is quite similar to a filter.

In our paper, we propose to characterize the expressive power of RNN in a quantitative way from the perspective of filtering. We consider a discrete filtering system as 1.

$$
\left\{ \begin{array}{l} x _ {k} = f \left(x _ {k - 1}\right) + g \left(x _ {k - 1}\right) w _ {k - 1}, \\ y _ {k} = h \left(x _ {k}\right) + v _ {k}, \end{array} \right. \tag {1}
$$

where the state  $x_{k}$  at time instant  $k$  is an  $n$ -dimensional vector,  $f$  is an  $n$ -dimensional vector-valued function,  $g$  is an  $n \times r$  matrix-valued function,  $\{w_{k}, k = 0,1,\dots\}$  is an  $r$ -dimensional white Gaussian process and  $w_{k} \sim \mathcal{N}(0,Q)$ , where  $Q$  is the covariance matrix of  $w_{k}$ .  $y_{k}$  is the  $m$ -dimensional observation (measurement) process,  $h$  is an  $m$ -dimensional vector-valued function,  $\{v_{k}, k = 1,\dots\}$  is an  $m$ -dimensional white Gaussian process and  $v_{k} \sim \mathcal{N}(0,R)$ , where  $R$  is the covariance matrix of  $v_{k}$ . And we assume that  $\{w_{k}, k = 0,1,\dots\}$ ,  $\{v_{k}, k = 1,\dots\}$  and the initial state  $x_{0}$  are jointly independent. We use  $Y_{k}$  to denote the sequence of observations up to time instant  $k$ , i.e.,

$$
Y _ {k} := \left\{y _ {1}, \dots , y _ {k} \right\}. \tag {2}
$$

Given the realization of the sequence of observations  $Y_{k}$ , the aim of filtering problem is to compute the optimal estimate of  $x_{k}$  conditioned on  $Y_{k}$ .

Not surprisingly, recurrent neural network has been proposed to do filtering. James Ting Ho Lo showed that Recurrent Multi-layer Perceptron can be used to synthesize optimal filter (Lo (1994)). However, Lo's approach is based on simply copying and storing the whole observation history in the hidden variables and thus require the time horizon to be finite, which is fundamentally limited. Besides Lo's work, many efforts have been made to connect RNN and dynamical system. Wilson & Finkel (2009) implemented a neural network based Kalman Filter (KF) but did not provide theoretical analysis on the approximation error. Parlos et al. (2001) proposed an algorithmic approach to do nonlinear filtering using recurrent neural network architecture but did not provide theoretical guarantee. Schäfer & Zimmermann (2006) shows that recurrent neural network is a universal approximator of dynamical system. However, they only consider the deterministic system dynamics and do not analyze the filtering relationship between two stochastic processes.

Compared to the existing work, we make the following specific contributions:

- Motivated by the similarity between RNN and filter, we propose to use the ability to approximate optimal filter to characterize the expressive power of RNN. Unlike existing work on expressive power of RNN (Schafer & Zimmermann (2006)) where only deterministic system is considered, we consider a stochastic dynamic system with noisy observations and analyze the capability of RNN to estimate unknown state.  
- We unify RNN-based filter into Bayesian filtering framework. In this framework, the hidden variables of RNN are interpreted as statistics of the observation history. And the evolution of hidden variables are interpreted as the evolution of statistics.  
- Based on the Bayesian filtering framework, we derive our main result: Recurrent Neural Networks (RNN) are universal approximators of a large class of optimal finite dimensional filters. That is to say, RNN estimator's asymptotic estimation error can be as close to minimum mean square error as desired. As an interesting special case, the widely used Kalman Filter can be synthesized by RNN. The consideration of asymptotic error differentiates us from existing work on expressive power of RNN (Schafer & Zimmermann (2006)).

# 2 PRELIMINARY: THE BAYESIAN FRAMEWORK OF FILTERING

We first introduce the definition of minimum mean square error estimate.

Definition 1 (Minimum Mean Square Error (MMSE) Estimate (Jazwinski (1970))). Let  $\hat{x}$  be an estimate of random variable  $x$  and  $L_{\mathrm{MSE}} \coloneqq (x - \hat{x})^T(x - \hat{x})$ . The estimate that minimizes  $\mathbb{E}[L_{\mathrm{MSE}}]$  is called the minimum mean square error estimate.

Theorem 1 (Theorem 5.3 in Jazwinski (1970)). Conditioned on the observation history  $Y_{k}$ , the minimum mean square error estimate of state  $x_{k}$  is the conditional mean  $\mathbb{E}[x_k|Y_k]$ .

Proof. We refer the reader to the proof in Section 5.2 of Jazwinski (1970).

![](images/029a6ec8a8f199c3812e72c3ce8acbc56e4a3344c5c000a3ae476459d283b5d0.jpg)

The Bayesian filtering consists of recursive prediction and update procedures (Jazwinski (1970)):

Prediction Step  $p(x_{k - 1}|Y_{k - 1}) \to p(x_k|Y_{k - 1})$ : Given the posterior distribution  $p(x_{k - 1}|Y_{k - 1})$  at instant  $k - 1$ , the prior distribution  $p(x_k|Y_{k - 1})$  of  $x_{k}$  satisfies the Fokker-Planck equation:

$$
p \left(x _ {k} \mid Y _ {k - 1}\right) = \int p \left(x _ {k} \mid x _ {k - 1}\right) p \left(x _ {k - 1} \mid Y _ {k - 1}\right) \mathrm {d} x _ {k - 1}; \tag {3}
$$

Updating Step  $p(x_{k}|Y_{k - 1})\to p(x_{k}|Y_{k})$ : Given prior distribution  $p(x_{k}|Y_{k - 1})$ , when the observation  $y_{k}$  at instant  $k$  arrives, the posterior distribution  $p(x_{k}|Y_{k})$  at instant  $k$  is given by equation 4,

$$
p \left(x _ {k} \mid Y _ {k}\right) = \frac {p \left(y _ {k} \mid x _ {k}\right) p \left(x _ {k} \mid Y _ {k - 1}\right)}{\int p \left(y _ {k} \mid x _ {k}\right) p \left(x _ {k} \mid Y _ {k - 1}\right) \mathrm {d} x _ {k}}. \tag {4}
$$

Then we can get the MMSE estimate by simply doing integration:

$$
\mathbb {E} \left[ x _ {k} \mid Y _ {k} \right] = \int p \left(x _ {k} \mid Y _ {k}\right) x _ {k} \mathrm {d} x _ {k}. \tag {5}
$$

To facilitate subsequent discussion, we make the following definition.

Definition 2 (Sufficient Statistic (Beneš (1981))). If the conditional distribution  $p(x_{k}|Y_{k})$  (or  $p(x_{k}|Y_{k - 1})$ ) can be fully determined by a vector-valued function  $s_k|_k$  (or  $s_{k|k - 1}$ ) of the observation sequence  $Y_{k}$  (or  $Y_{k - 1}$ ), then we say  $s_{k|k}$  (or  $s_{k|k - 1}$ ) is a sufficient statistic for  $p(x_{k}|Y_{k})$  (or  $p(x_{k}|Y_{k - 1})$ ).<sup>1</sup>

Because sufficient statistic  $s_{k|k}$  fully determines the posterior distribution  $p(x_k | Y_k)$  and MMSE estimate  $\mathbb{E}(x_k | Y_k)$  is a functional of  $p(x_k | Y_k)$ , there exists a function  $\gamma$  that maps  $s_{k|k}$  to  $\mathbb{E}[x_k | Y_k]$ , i.e.,

$$
\mathbb {E} \left[ x _ {k} \mid Y _ {k} \right] = \gamma \left(s _ {k \mid k}\right). \tag {6}
$$

# 3 RNN BASED FILTER'S ARCHITECTURE

Motivated by the Bayesian framework of filtering, we propose the RNN based filter's architecture as shown in Fig. 1.

Our RNN based filter's architecture, Bayesian Filter Net (BFN), consists of three parts: prediction network, update network and estimation network. To mimic the prediction step in Bayesian filtering, we use the prediction network to map the posterior distribution representation vector to a prior distribution representation vector. To mimic the update step in Bayesian filtering, we then use an update network to update the prior state distribution representation vector and the observation to get the posterior state distribution representation vector. Finally, we use an estimation network to map the current posterior state distribution representation vector to current estimation  $\hat{x}_k$ . We will see in the subsequent discussions, the so-called representation vector or hidden variables indeed can be interpreted as statistics of the underlying conditional distribution.

# 4 RNN BASED FILTER IS UNIVERSAL

We now show that our proposed RNN based filter is universal in that it can approximate a large class of optimal finite dimensional filters to any asymptotic accuracy we desire. We summarize our insight into the diagram 2.

![](images/a764cdd3745122e3c99e81448fa3a70644a4a3b2e5b27f9b54e47afdeaf6f449.jpg)  
Figure 1: RNN based filter's architecture: Bayesian Filter Net (BFN).

![](images/429f020d7f771fecf71f6ef4462386abe0562d3506f95e3c52c729a01a4afaed.jpg)  
Figure 2: Our approach's illustrative diagram

As shown in the diagram 2, we model the conditional prior probability  $p(x_{k}|Y_{k - 1})$  (posterior probability  $p(x_{k}|Y_{k})$  resp.) of the state at step  $k$  by finite dimensional prior statistics  $s_{k|k - 1}$  (posterior statistics  $s_{k|k}$  resp.). In finite dimensional filter case (Beneš (1981; 1985); Daum (1987)), there exist finite dimensional statistics that are sufficient, that is to say, the evolution of conditional probability can be fully captured by the evolution of a finite dimensional vector. We denote the evolution function in updating step by  $\varphi$  and the evolution function in prediction step by  $\phi$ . And further, after modelling the the probability distribution as finite dimensional statistics, we use two neural networks to approximate the evolution of them. And to get the final estimate of the state, we use another neural network to approximate the map from the statistics to the optimal estimation.

For the RNN based filter, one can naturally ask:

1. Will the neural network approximation error accumulate and blow up when time goes to infinity and make RNN asymptotically not work at all?  
2. How general is this approach? What filter can RNN approximate?

We give the answers in this section.

# 4.1 KALMAN FILTER (KF) CAN BE SYNTHESIZED BY RNN

When system equation 1 is linear and satisfies Gaussian noise assumption as shown in equation 7, it is well known that the filtering problem can be optimally solved by Kalman Filter (KF) (Kalman (1960)). (See more details in our Appendix A.2.)

$$
\left\{ \begin{array}{l} x _ {k} = F x _ {k - 1} + G w _ {k - 1} \\ y _ {k} = H x _ {k} + v _ {k}, \end{array} \right. \tag {7}
$$

where  $F, G, H$  are constant matrices with proper dimensions, the initial state  $x_0$  is Gaussian, and  $\{w_k, k = 0,1,\dots\}$  and  $\{v_k, k = 1,\dots\}$  are two independent white Gaussian sequences that are also independent of the initial state  $x_0$ .

It can be known that the prior and posterior distributions are Gaussian and fully characterized by the sufficient statistics  $s$  composed of mean and covariance matrix. (See more details in our Appendix A.2.)

$$
\left\{ \begin{array}{r l} s _ {k \mid k - 1} & := [ m _ {k \mid k - 1}, \operatorname {v e c} ^ {T} \left(P _ {k \mid k - 1}\right) ] ^ {T}, \\ s _ {k \mid k} & := [ m _ {k \mid k}, \operatorname {v e c} ^ {T} \left(P _ {k \mid k}\right) ] ^ {T}, \end{array} \right. \tag {8}
$$

where  $m_{k|k-1}$  and  $P_{k|k-1}$  are the conditional mean and covariance in step  $k$  conditioned on  $Y_{k-1}$  and  $m_{k|k}$  and  $P_{k|k}$  are the mean and covariance in step  $k$  conditioned on  $Y_{k}$ , and  $\mathrm{vec}(\circ_{n_1 \times n_2})$  is the  $n_1 n_2 \times 1$  column vector obtained by stacking the columns of the matrix  $\circ$  on top of one another.  $s_{k|k}$  ( $s_{k|k-1}$  resp.) is the theoretical statistic that determines the conditional probability distribution of the state  $x_k$  conditioned on the observation history  $Y_k$  ( $Y_{k-1}$  resp.) and evolves according to some function  $\varphi$  and  $\phi$ . (See more details in our Appendix A.2.) We also know that there exists some function  $\gamma$  that maps  $s_{k|k}$  to MMSE estimate  $\mathbb{E}[x_k | Y_k]$ . Thus we have,

$$
s _ {k \mid k} = \varphi \left(s _ {k \mid k - 1}, y _ {k}\right), s _ {k + 1 \mid k} = \phi \left(s _ {k \mid k}\right), \mathbb {E} \left[ x _ {k} \mid Y _ {k} \right] = \gamma \left(s _ {k \mid k}\right). \tag {9}
$$

We use function  $\tilde{\varphi}$  generated by a deep (feedforward) neural network (i.e. update network) to approximate  $\varphi$ , and use  $\tilde{\phi}$  generated by another DNN (i.e. prediction network) to approximate  $\phi$ . And the numerical statistics computed by RNN are denoted as  $\tilde{s}_{k|k}$  and  $\tilde{s}_{k+1|k}$ , i.e.,

$$
\tilde {s} _ {k \mid k} = \tilde {\varphi} (\tilde {s} _ {k \mid k - 1}, y _ {k}), \tilde {s} _ {k + 1 \mid k} = \tilde {\phi} (\tilde {s} _ {k \mid k}). \tag {10}
$$

And we use function  $\tilde{\gamma}$  generated by the third deep feedforward neural network (i.e. estimation network) to approximate  $\gamma$  in equation 6, i.e.,

$$
\hat {x} _ {k \mid k} = \tilde {\gamma} \left(\tilde {s} _ {k \mid k}\right). \tag {11}
$$

Note that the probability space  $(\Omega, \mathcal{F}, \mathbb{P})$  with finite second moment, with inner product  $\langle x, y \rangle = \mathbb{E}[x^T y]$  and norm  $\|x\| \coloneqq \mathbb{E}^{1/2}[x^T x]$  is a Hilbert space, denoted as  $L^2(\Omega, \mathcal{F}, \mathbb{P})$ . We first state the universal approximation theorem of feedforward neural network before we proceed to show our results.

Theorem 2 (Universal Approximation Theorem (Hornik et al. (1989))). For any given compact subset  $K \subset R^n$ , any given continuous function  $f$  defined on  $K$  and any given accuracy degree  $\epsilon > 0$ , there exists a function  $g$  represented by a single-hidden-layer neural network with non-constant and bounded activation function such that  $\max_{x \in K} |f(x) - g(x)| < \epsilon$ .

Proof. It is a natural corollary of the Thm. 2.1 in Hornik et al. (1989).

![](images/7d149fa11c85b7918e236087dda26262c26447a1acc5030b6f9e18f7d31b2eb7.jpg)

Define  $e_{k|k} \coloneqq \| s_{k|k} - \tilde{s}_{k|k} \|$ , which represents the cumulative error caused by the approximation error of  $\tilde{\phi}$  and  $\tilde{\varphi}$ . Similarly, we define  $e_{k|k-1} \coloneqq \| s_{k|k-1} - \tilde{s}_{k|k-1} \|$ . In the following theorem, we shall give the condition which ensures the cumulative error will not blow up as time  $k$  approaches  $\infty$ .

Before we proceed to show our main result, we first establish a key lemma, Lem. 2. We make two assumptions on the system we'll consider..

Assumption 1. We assume that the linear dynamic system of the state in equation 7 is stable in mean square sense (Samuels (1959)), i.e.,

$$
\varlimsup_ {k \rightarrow \infty} \| x _ {k} \| \leq M, \tag {12}
$$

where  $M$  is a finite constant.

Assumption 2. The dynamical system equation 7 is uniformly completely observable and uniformly completely controllable.

The definitions of uniformly completely observable and uniformly completely controllable can be found in section 7.5 of Jazwinski (1970). We then state a lemma on the boundedness of conditional covariance.

Lemma 1 (Lemma 7.1 in Jazwinski (1970)). If Assumption 2 is satisfied and  $P_{0|0} \succcurlyeq 0^3$ , then  $P_{k|k}$  is uniformly bounded from above for all  $k \geq N$ ,

$$
P _ {k \mid k} \preccurlyeq \left(\frac {1 + \alpha \beta}{\alpha}\right) I, k \geq N, \tag {13}
$$

where  $N$  is a positive integer,  $I$  is the  $n \times n$  identity matrix and  $\alpha$ ,  $\beta$  are positive constants.

Based on Lemma 1, Assumption 1 and Assumption 2, we give the key Lemma 2.

Lemma 2. In the discrete linear system equation 7, suppose the Assumption 1 and Assumption 2 are satisfied, then for any given  $\epsilon >0$  there exists a compact subset  $K\subset \mathrm{R}^{\dim (s_{k|k})}$  such that the statistics computed by  $KF$ $s_{k|k - 1},s_{k|k}$  and the statistics  $\tilde{s}_{k|k - 1},\tilde{s}_{k|k}$  computed by RNN based filter with nonconstant and bounded activation function satisfy  $\left\| s_{k|k - 1}\mathbb{1}_{s_{k|k - 1}\notin K}\right\| < \epsilon ,\left\| s_{k|k}\mathbb{1}_{s_{k|k}\not\in K}\right\| < \epsilon ,$ $\left\| \tilde{s}_{k|k - 1}\mathbb{1}_{\tilde{s}_k|k - 1\notin K}\right\| < \epsilon$  and  $\left\| \tilde{s}_{k|k}\mathbb{1}_{\tilde{s}_{k|k}\not\in K}\right\| < \epsilon$  where  $\mathbb{1}_A$  is an indicator function.

Proof. Proof can be found in our appendix A.3.

![](images/0994690c3d5c4fcfa69e5e000efb6fd3f27cceb6c0c73ed93a5226abcc638ccc.jpg)

We then derive our main result.

Theorem 3. Assume  $s_{k|k}, k \geq 0$  are the theoretical statistics evolving according to equation 9 and  $\tilde{s}_{k|k}, k \geq 0$  are the real statistics computed by our RNN-based filter evolving according to equation 10. Suppose the Assumption 1 and Assumption 2 are satisfied. Furthermore, we need to assume functions  $\varphi, \phi, \gamma$  are Lipschitz, i.e., for any  $s_1, s_2$ ,

$$
\begin{array}{l} \left\| \varphi (s _ {1}, y) - \varphi (s _ {2}, y) \right\| \leq C _ {\varphi} \| s _ {1} - s _ {2} \|, \\ \left\| \phi \left(s _ {1}\right) - \phi \left(s _ {2}\right) \right\| \leq C _ {\phi} \| s _ {1} - s _ {2} \|, \tag {14} \\ \left\| \gamma \left(s _ {1}\right) - \gamma \left(s _ {2}\right) \right\| \leq C _ {\gamma} \| s _ {1} - s _ {2} \|, \\ \end{array}
$$

where  $C_{\varphi}$  and  $C_{\phi}$  are Lipschitz constants. If  $C_{\varphi}$  and  $C_{\phi}$  satisfy  $|C_{\varphi}C_{\phi}| < 1$ , then for any  $\epsilon > 0$ , there exists an RNN based filter (with non-constant and bounded activation function) such that

$$
\lim  _ {k \rightarrow \infty} \sup  _ {k \rightarrow \infty} e _ {k | k} = \lim  _ {k \rightarrow \infty} \sup  _ {k \rightarrow \infty} \left\| s _ {k | k} - \tilde {s} _ {k | k} \right\| <   \epsilon . \tag {15}
$$

Furthermore, we have

$$
\lim  _ {k \rightarrow \infty} \left\| \hat {x} _ {k | k} - \mathbb {E} \left[ x _ {k} \mid Y _ {k} \right]\right\| <   \epsilon . \tag {16}
$$

Proof. For any  $\delta > 0$ , we have the following. By Lem. 3, there exists a compact ball  $K = \mathrm{B}(0,r) \subset \mathbf{R}^{\dim(s_{k|k})}$ , such that  $\left\|s_{k|k-1}\mathbb{1}_{s_{k|k-1}\notin K}\right\| < \delta$ ,  $\left\|s_{k|k}\mathbb{1}_{s_{k|k}\notin K}\right\| < \delta$ ,  $\left\|\tilde{s}_{k|k-1}\mathbb{1}_{\tilde{s}_{k|k-1}\notin K}\right\| < \delta$  and  $\left\|\tilde{s}_{k|k}\mathbb{1}_{\tilde{s}_{k|k}\notin K}\right\| < \delta$ . By Theorem 2, given any small  $\delta_{\varphi}, \delta_{\phi} \in \mathbb{R}^{+}$  and  $\delta_{\gamma} \in \mathbb{R}^{+}$ , there exist two functions  $\tilde{\varphi}, \tilde{\phi}$  which are represented by the DNN, such that

$$
\left\| \varphi - \tilde {\varphi} \right\| _ {\infty} ^ {K} \leq \delta_ {\varphi}, \left\| \phi - \tilde {\phi} \right\| _ {\infty} ^ {K} \leq \delta_ {\phi}, \left\| \gamma - \tilde {\gamma} \right\| _ {\infty} ^ {K} \leq \delta_ {\gamma}. \tag {17}
$$

where  $\| h\|_{\infty}^{K}\coloneqq \max_{x\in K}|h(x)|$  . And without loss of generality, we set  $\phi (0) = \tilde{\phi} (0)$ $\varphi (0) = \tilde{\varphi} (0)$  and  $\gamma (0) = \widetilde{\gamma} (0)$  . In the prediction step, based on the evolution equations equation 9 and equation 10, we have

$$
\begin{array}{l} e _ {k | k - 1} = \left\| \left(s _ {k | k - 1} - \tilde {s} _ {k | k - 1}\right) \right\| = \left\| \phi \left(s _ {k - 1 | k - 1}\right) - \tilde {\phi} \left(\tilde {s} _ {k - 1 | k - 1}\right) \right\| \\ \leq \left\| \phi \left(s _ {k - 1 \mid k - 1}\right) - \phi \left(\tilde {s} _ {k - 1 \mid k - 1}\right) \right\| + \left\| \phi \left(\tilde {s} _ {k - 1 \mid k - 1}\right) - \tilde {\phi} \left(\tilde {s} _ {k - 1 \mid k - 1}\right) \right\| \\ \leq \left\| \phi \left(s _ {k - 1 \mid k - 1}\right) - \phi \left(\tilde {s} _ {k - 1 \mid k - 1}\right) \right\| + \left\| \left(\phi \left(\tilde {s} _ {k - 1 \mid k - 1}\right) - \tilde {\phi} \left(\tilde {s} _ {k - 1 \mid k - 1}\right)\right) \mathbb {1} _ {\tilde {s} _ {k - 1 \mid k - 1} \in K} \right\| \tag {18} \\ + \left\| \left(\phi \left(\tilde {s} _ {k - 1 \mid k - 1}\right) - \tilde {\phi} \left(\tilde {s} _ {k - 1 \mid k - 1}\right)\right) \mathbb {1} _ {\tilde {s} _ {k - 1 \mid k - 1} \notin K} \right\| \\ \leq C _ {\phi} e _ {k - 1 | k - 1} + \delta_ {\phi} + (C _ {\phi} + C _ {\tilde {\phi}}) \delta , \\ \end{array}
$$

where the last inequality follows from equation 14 and equation 17 and  $C_{\tilde{\phi}}$  is the Lipschitz constant of  $\tilde{\phi}$ . We let  $\delta_{\phi}^{\prime} \coloneqq \delta_{\phi} + (C_{\phi} + C_{\tilde{\phi}})\delta$ .

Similarly, in the updating step, we have  $e_{k|k} \leq C_{\varphi} e_{k|k-1} + \delta_{\varphi}'$ , where  $\delta_{\varphi}' \coloneqq \delta_{\varphi} + (C_{\varphi} + C_{\tilde{\varphi}}) \delta$ . Combining this and equation 18, we obtain

$$
e _ {k \mid k} \leq C _ {\varphi} e _ {k \mid k - 1} + \delta_ {\varphi} ^ {\prime} \leq \left(C _ {\varphi} C _ {\phi}\right) e _ {k - 1 \mid k - 1} + \left(C _ {\varphi} \delta_ {\phi} ^ {\prime} + \delta_ {\varphi} ^ {\prime}\right). \tag {19}
$$

Using equation 19 repeatedly, it follows that

$$
e _ {k \mid k} \leq \left(C _ {\varphi} C _ {\phi}\right) ^ {k} e _ {0 \mid 0} + \left(C _ {\varphi} \delta_ {\phi} ^ {\prime} + \delta_ {\varphi} ^ {\prime}\right) \frac {\left(C _ {\varphi} C _ {\phi}\right) ^ {k} - 1}{C _ {\varphi} C _ {\phi} - 1}. \tag {20}
$$

Thus  $\lim_{k\to +\infty}e_{k|k}\leq \left(C_{\varphi}\left(\delta_{\phi} + (C_{\phi} + C_{\tilde{\phi}})\delta\right) + \delta_{\varphi} + (C_{\varphi} + C_{\tilde{\varphi}})\delta\right)\frac{1}{1 - C_{\varphi}C_{\phi}}$  as  $k\to \infty$  once the condition  $|C_{\varphi}C_{\phi}| < 1$  holds. We choose small enough  $\delta_{\phi},\delta_{\varphi}$  and  $\delta$  such that  $\left(C_{\varphi}\left(\delta_{\phi} + (C_{\phi} + C_{\tilde{\phi}})\delta\right) + \delta_{\varphi} + (C_{\varphi} + C_{\tilde{\varphi}})\delta\right)\frac{1}{1 - C_{\varphi}C_{\phi}} < \epsilon$ . Then we get 15. Now we prove 16.

$$
\begin{array}{l} \left\| \hat {x} _ {k | k} - \mathbb {E} \left[ x _ {k} \mid Y _ {k} \right] \right\| = \left\| \tilde {\gamma} \left(\tilde {s} _ {k | k}\right) - \gamma \left(s _ {k | k}\right) \right\| \leq \left\| \gamma \left(s _ {k | k}\right) - \gamma \left(\tilde {s} _ {k | k}\right) \right\| + \left\| \gamma \left(\tilde {s} _ {k | k}\right) - \tilde {\gamma} \left(\tilde {s} _ {k | k}\right) \right\| \\ \leq \left\| \gamma \left(s _ {k \mid k}\right) - \gamma \left(\tilde {s} _ {k \mid k}\right) \right\| + \left\| \left(\gamma \left(\tilde {s} _ {k \mid k}\right) - \tilde {\gamma} \left(\tilde {s} _ {k \mid k}\right)\right) \mathbb {1} _ {\tilde {s} _ {k - 1 \mid k - 1} \in K} \right\| + \left\| \left(\gamma \left(\tilde {s} _ {k \mid k}\right) - \tilde {\gamma} \left(\tilde {s} _ {k \mid k}\right)\right) \mathbb {1} _ {\tilde {s} _ {k - 1 \mid k - 1} \not \in K} \right\| \\ \stackrel {\star 1} {\leq} C _ {\gamma} e _ {k \mid k} + \delta_ {\gamma} + \left(C _ {\gamma} + C _ {\tilde {\gamma}}\right) \delta \stackrel {\star 2} {\leq} C _ {\gamma} \left(C _ {\varphi} C _ {\phi}\right) ^ {k} e _ {0 \mid 0} + C _ {\gamma} \left(C _ {\varphi} \delta_ {\phi} ^ {\prime} + \delta_ {\varphi} ^ {\prime}\right) \frac {\left(C _ {\varphi} C _ {\phi}\right) ^ {k} - 1}{C _ {\varphi} C _ {\phi} - 1} + \delta_ {\gamma} + \left(C _ {\gamma} + C _ {\tilde {\gamma}}\right) \delta \tag {21} \\ \end{array}
$$

where the inequality  $\star_{1}$  follows equation 14 and equation 17, the inequality  $\star_{2}$  follows equation 54, and  $C_{\tilde{\gamma}}$  is the Lipschitz constant of  $\tilde{\gamma}$ . Thus  $\limsup_{k\to +\infty}\left\| \hat{x}_{k|k} - \mathbb{E}[x_k|Y_k]\right\|\leq C_\gamma (C_\varphi \delta_\phi^j + \delta_\varphi ')(-C_\varphi C_\phi +1)^{-1} + \delta_\gamma +(C_\gamma +C_{\tilde{\gamma}})\delta$ . Again we can choose small enough  $\delta_{\phi},\delta_{\varphi},\delta_{\gamma}$  and  $\delta$  such that  $C_\gamma (C_\varphi \delta_\phi ' + \delta_\varphi ')(-C_\varphi C_\phi +1)^{-1} + \delta_\gamma +(C_\gamma +C_{\tilde{\gamma}})\delta < \epsilon$ . Then we obtain the desired 16.

An example satisfying all the assumptions of Thm. 3 can be found in our Appendix A.4. We also remark that in our proof, we implicitly require that the Lipschitz constants of  $\tilde{\phi},\tilde{\varphi},\tilde{\gamma}$  are uniformly upper bounded. (See more details in our Appendix A.6) Thm. 3 highlights that the optimal filter in linear system with Gaussian noise, Kalman Filter, can be synthesized by RNN. And RNN based filter's asymptotic error can be as small as wanted under some Lipschitz conditions. That is to say, RNN is an approximator of Kalman filter.

# 4.2 RNN BASED FILTER IS A UNIVERSAL APPROXIMATOR OF OPTIMAL FINITE DIMENSIONAL FILTER

Thm. 3 shows that Kalman Filter (KF) can be synthesized by RNN. In this section, we'll try to answer the question "How general is the RNN based filter?" and extend the result into a more general case. We'll show that any optimal finite dimensional filter can be universally approximated by RNN under some mild conditions. For a general system with noisy observation as shown in equation 1, once the conditional distribution  $p(x_{k}|Y_{k})$  is obtained, the filtering problem is solved. However, we usually need to solve an infinite number of ordinary differential equations (ODE) in order to solve  $p(x_{k}|Y_{k})$ . If the distribution  $p(x_{k}|Y_{k})$  admits a finite dimensional sufficient statistics, then we only need to solve a finite number of ODE (Chen (2003)) and we call such filter finite dimensional filter. Finite dimensional filter has been an active research area after the seminal work (Benes (1981; 1985)) of Beneš. It's a large class of filters. Some nontrivial finite dimensional nonlinear filter examples can be found in Daum (1986); Ferrante & Runggaldier (1990); GUnther (1981); Levine & Marino (1986).

Similarly, we use vector  $S_{k|k}$  to denote the finite dimensional sufficient statistics of the posterior distribution  $p(x_k | Y_k)$  and  $S_{k|k-1}$  to denote the finite dimensional sufficient statistics of the prior distribution  $p(x_k | Y_{k-1})$ . The evolution functions of the statistics are denoted as  $\Phi$  and  $\Psi$ , and the map from  $S_{k|k}$  to conditional mean  $\mathbb{E}(x_k | Y_k)$  is denoted as  $\Gamma$ , i.e.,

$$
S _ {k \mid k - 1} = \Phi \left(S _ {k - 1 \mid k - 1}\right), S _ {k \mid k} = \Psi \left(S _ {k \mid k - 1}, y _ {k}\right), \mathbb {E} \left(x _ {k} \mid Y _ {k}\right) = \Gamma \left(S _ {k \mid k}\right). \tag {22}
$$

Similarly, in our proposed neural networks, we use DNN generated function  $\tilde{\Phi}$  (prediction network) to approximate  $\Phi$ , use another DNN generated  $\tilde{\Psi}$  (update network) to approximate  $\Psi$ , and use the third DNN generated function  $\tilde{\Gamma}$  to approximate  $\Gamma$ . And the numerical statistics computed by RNN are denoted as  $\tilde{S}_{k|k}$  and  $\tilde{S}_{k|k-1}$ , i.e.,

$$
\tilde {S} _ {k \mid k - 1} = \tilde {\Phi} (\tilde {S} _ {k - 1 \mid k - 1}), \tilde {S} _ {k \mid k} = \tilde {\Psi} (\tilde {S} _ {k \mid k - 1}, y _ {k}), \hat {x} _ {k \mid k} = \tilde {\Gamma} (\tilde {S} _ {k \mid k}). \tag {23}
$$

We also need the following assumption.

Assumption 3. We assume that for any given  $\epsilon >0$  there exists a compact subset  $K\subset \mathrm{R}^{\dim (S_{k|k})}$  such that the statistics computed by the optiaml finite dimensional filter  $S_{k|k - 1}$  and  $S_{k|k}$  satisfy  $\left\| S_{k|k - 1}\mathbb{1}_{S_{k|k - 1}}\notin K\right\| < \epsilon$ , and  $\left\| S_{k|k}\mathbb{1}_{S_{k|k}}\notin K\right\| < \epsilon$ .

We can see equation 7 is the special case of system equation 1, and it satisfies the Assumption 3 according to Lemma 2. We further have Lem. 3 and Thm. 4.

Lemma 3. In the discrete system equation 1, for any given  $\epsilon >0$  there exists a compact subset  $K\subset \mathrm{R}^{\dim (S_{k|k})}$  such that the statistics  $\tilde{S}_{k|k - 1},\tilde{S}_{k|k}$  computed by RNN based filter with non-constant and bounded activation function satisfy  $\left\| \tilde{S}_{k|k - 1}\mathbb{1}_{\tilde{S}_{k|k - 1}\notin K}\right\| <  \epsilon$  and  $\left\| \tilde{S}_{k|k}\mathbb{1}_{\tilde{S}_{k|k}\notin K}\right\| <  \epsilon$  , where  $\mathbb{1}_A$  is indicator function.

Proof. The proof is similar to the step 2 of the proof of Lem. 2.

![](images/bae445548b5ab18d2e50ac64a069d618b61c0631970a32b079f1641b62c43559.jpg)

Theorem 4. Consider a discrete filtering system equation 1 with optimal finite dimensional filter and suppose  $S_{k|k}$ ,  $k \geq 0$  are the theoretical statistics evolving according to equation 22 and  $\tilde{S}_{k|k}$ ,  $k \geq 0$  are the statistics generated by our RNN based filter and evolving according to equation 23. Suppose the Assumption 3 is satisfied. Furthermore, we need to assume that functions  $\Phi$  and  $\Psi$  are Lipschitz, i.e., for any  $S_1, S_2$ ,

$$
\begin{array}{l} \| \Psi (S _ {1}, y) - \Psi (S _ {2}, y) \| \leq C _ {\Psi} \| S _ {1} - S _ {2} \|, \\ \left\| \Phi \left(S _ {1}\right) - \Phi \left(S _ {2}\right) \right\| \leq C _ {\Phi} \| S _ {1} - S _ {2} \|, \tag {24} \\ \left\| \Gamma (S _ {1}) - \Gamma (S _ {2}) \right\| \leq C _ {\Gamma} \| S _ {1} - S _ {2} \| \\ \end{array}
$$

where  $C_{\Psi}$  and  $C_{\Phi}$  are Lipschitz constants. If  $C_{\Psi}$  and  $C_{\Phi}$  satisfy  $|C_{\Psi}C_{\Phi}| < 1$ , then for any  $\epsilon > 0$ , there exists a RNN based filter with non-constant and bounded activation function such that

$$
\lim  _ {k \rightarrow \infty} \left\| S _ {k | k} - \tilde {S} _ {k | k} \right\| <   \epsilon . \tag {25}
$$

Furthermore, we have

$$
\operatorname * {l i m s u p} _ {k \rightarrow \infty} \left\| \hat {x} _ {k \mid k} - \mathbb {E} \left[ x _ {k} \mid Y _ {k} \right]\right\| <   \epsilon . \tag {26}
$$

Proof. The procedure is similar to the proof of Theorem 3.

![](images/f6103a4fbfedee9a1d82198fc98ae094e8483373181c6a02c932d91319e78fa6.jpg)

Thm. 4 highlights that RNN based filter can not only approximate Kalman filter, but any optimal finite dimensional filter under some Lipschitz conditions. Therefore, RNN's expressive power is characterized as the universal filtering property.

# 5 CONCLUSION

In our paper, we try to characterize the expressive power of RNN from the filtering perspective. We unify the recurrent neural network into Bayesian filtering framework and show that recurrent neural network is a universal approximator of optimal finite dimensional filters under some Lipschitz conditions. As an interesting special case, the widely used Kalman filter can be synthesized by RNN. Understanding the expressive power of RNN based filter in more general nonlinear filtering cases (with no finite dimensional sufficient statistics) can be a very interesting future direction.

# REFERENCES

Cem Anil, James Lucas, and Roger Grosse. Sorting out lipschitz function approximation. arXiv preprint arXiv:1811.05381, 2018.  
V. E. Beneš. Exact finite-dimensional filters for certain diffusions with nonlinear drift. Stochastics-an International Journal of Probability & Stochastic Processes, 5(1-2):65-92, 1981.  
V. E. Benes. New exact nonlinear filters with large lie algebras. Systems & Control Letters, 5(4): 217-221, 1985.  
Zhe Chen. Bayesian filtering: from Kalman filters to particle filters, and beyond. Statistics: A Journal of Theoretical and Applied Statistics, 182(1):1-69, 2003.  
F. Daum. Solution of the zakai equation by separation of variables. IEEE Transactions on Automatic Control, 32(10):941-943, 1987.  
Frederick Daum. Exact finite-dimensional nonlinear filters. IEEE Transactions on Automatic Control, 31(7):616-622, 1986.  
Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1110-1118, 2015.  
Juntao Fei and Cheng Lu. Adaptive sliding mode control of dynamic systems using double loop recurrent neural network structure. IEEE Transactions on Neural Networks and Learning Systems, 29(4):1275-1286, 2017.  
Marco Ferrante and Wolfgang J Runggaldier. On necessary conditions for the existence of finite-dimensional filters in discrete time. Systems & control letters, 14(1):63-69, 1990.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In 2013 IEEE international conference on acoustics, speech and signal processing, pp. 6645-6649. IEEE, 2013.  
Sawitzki G'Unther. Finite dimensional filter systems in discrete time. Stochastics: An International Journal of Probability and Stochastic Processes, 5(1-2):107-114, 1981.  
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989.  
Andrew H Jazwinski. Stochastic processes and filtering theory. Academic Press, New York and London, 1970.  
Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. Journal of basic Engineering, 82(1):35-45, 1960.  
J Levine and R Marino. Nonlinear system immersion, observers and finite-dimensional filters. Systems & Control Letters, 7(2):133-142, 1986.  
J Ting-Ho Lo. Synthetic approach to optimal filtering. IEEE Transactions on Neural Networks, 5(5): 803-811, 1994.  
Zhou Lu, Hongming Pu, Feicheng Wang, Zhiqiang Hu, and Liwei Wang. The expressive power of neural networks: A view from the width. In Advances in Neural Information Processing Systems, pp. 6231-6239, 2017.  
Xiaolei Ma, Zhimin Tao, Yinhai Wang, Haiyang Yu, and Yunpeng Wang. Long short-term memory neural network for traffic speed prediction using remote microwave sensor data. Transportation Research Part C: Emerging Technologies, 54:187-197, 2015.  
Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. Recurrent neural network based language model. In Eleventh annual conference of the international speech communication association, 2010.

Jooyoung Park and Irwin W Sandberg. Universal approximation using radial-basis-function networks. Neural computation, 3(2):246-257, 1991.  
Alexander G Parlos, Sunil K Menon, and A Atiya. An algorithmic approach to adaptive state filtering using recurrent neural networks. IEEE Transactions on Neural Networks, 12(6):1411-1432, 2001.  
J Samuels. On the mean square stability of random linear systems. IRE Transactions on Circuit Theory, 6(5):248-259, 1959.  
Anton Maximilian Schäfer and Hans Georg Zimmermann. Recurrent neural networks are universal approximators. In International Conference on Artificial Neural Networks, pp. 632-640. Springer, 2006.  
Robert Wilson and Leif Finkel. A neural implementation of the kalman filter. In Advances in neural information processing systems, pp. 2062-2070, 2009.
