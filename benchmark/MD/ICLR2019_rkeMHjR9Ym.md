# STOCHASTIC GRADIENT DESCENT LEARNS STATE EQUATIONS WITH NONLINEAR ACTIVATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study discrete time dynamical systems governed by the state equation  $\pmb{h}_{t + 1} = \phi (\pmb {A}\pmb {h}_t + \pmb {B}\pmb {u}_t)$ . Here  $\pmb{A},\pmb{B}$  are weight matrices,  $\phi$  is an activation function, and  $\pmb{u}_t$  is the input data. This relation is the backbone of recurrent neural networks (e.g. LSTMs) which have broad applications in sequential learning tasks. We utilize stochastic gradient descent to learn the weight matrices from a finite input/state trajectory  $\{\pmb {u}_t,\pmb {h}_t\}_{t = 0}^N$ . We prove that SGD estimate linearly converges to the ground truth weights while using near-optimal sample size. Our results apply to increasing activations whose derivatives are bounded away from zero. The analysis is based on i) an SGD convergence result with nonlinear activations and ii) careful statistical characterization of the state vector. Numerical experiments verify the fast convergence of SGD on ReLU and leaky ReLU in consistence with our theory.

# 1 INTRODUCTION

A wide range of problems involve sequential data with a natural temporal ordering. Examples include natural language processing, time series prediction, system identification, and control design, among others. State-of-the-art algorithms for sequential problems often stem from dynamical systems theory and are tailored to learn from temporally dependent data. Linear models and algorithms; such as Kalman filter, PID controller, and linear dynamical systems, have a long history and are utilized in control theory since 1960's with great success (Brown et al. (1992); Ho & Kalman (1966); Åström & Hägglund (1995)). More recently, nonlinear models such as recurrent neural networks (RNN) found applications in complex tasks such as machine translation and speech recognition (Bahdanau et al. (2014); Graves et al. (2013); Hochreiter & Schmidhuber (1997)). Unlike feedforward neural networks, RNNs are dynamical systems that use their internal state to process inputs. The goal of this work is to shed light on the inner workings of RNNs from a theoretical point of view. In particular, we focus on the RNN state equation which is characterized by a nonlinear activation function  $\phi$ , state weight matrix  $A$ , and input weight matrix  $B$  as follows

$$
\boldsymbol {h} _ {t + 1} = \phi (\boldsymbol {A} \boldsymbol {h} _ {t} + \boldsymbol {B} \boldsymbol {u} _ {t}), \tag {1.1}
$$

Here  $h_t$  is the state vector and  $u_t$  is the input data at timestamp  $t$ . This equation is the source of dynamic behavior of RNNs and distinguishes RNN from feedforward networks. The weight matrices  $A$  and  $B$  govern the dynamics of the state equation and are inferred from data. We will explore the statistical and computational efficiency of stochastic gradient descent (SGD) for learning these weight matrices.

Contributions: Suppose we are given a finite trajectory of input/state pairs  $(\boldsymbol{u}_t, \boldsymbol{h}_t)_{t=0}^N$  generated from the state equation (1.1). We consider a least-squares regression obtained from  $N$  equations; with inputs  $(\boldsymbol{u}_t, \boldsymbol{h}_t)_{t=1}^N$  and outputs  $(\boldsymbol{h}_{t+1})_{t=1}^N$ . For a class of activation functions including leaky ReLU and for stable systems<sup>1</sup>, we show that SGD linearly converges to the ground truth weight matrices while requiring near-optimal trajectory length  $N$ . In particular, the required sample size is  $\mathcal{O}(n + p)$  where  $n$  and  $p$  are the dimensions of the state and input vectors respectively. Our results are extended to unstable systems when the samples are collected from multiple independent RNN trajectories rather than a single trajectory. Our results apply to increasing activation functions whose derivatives are bounded away from zero; which includes leaky ReLU. Numerical experiments on

ReLU and leaky ReLU corroborate our theory and demonstrate that SGD converges faster as the activation slope increases. To obtain our results, we i) characterize the statistical properties of the state vector (e.g. well-conditioned covariance) and ii) derive a novel SGD convergence result with nonlinear activations; which may be of independent interest. As a whole, this paper provides a step towards foundational understanding of RNN training via SGD.

# 1.1 RELATED WORK

Our work is related to the recent optimization and statistics literature on linear dynamical systems (LDS) and neural networks.

Linear dynamical systems: The state-equation (1.1) reduces to a LDS when  $\phi$  is the linear activation  $(\phi(x) = x)$ . Identifying the weight matrices is a core problem in linear system identification and is related to the optimal control problem (e.g. linear quadratic regulator) with unknown system dynamics. While these problems are studied since 1950's (Ljung (1998; 1987); Åström & Eykhoff (1971)), our work is closer to the recent literature that provides data-dependent bounds and characterize the non-asymptotic learning performance. Recht and coauthors have a series of papers exploring optimal control problem (Simchowitz et al. (2018); Tu et al. (2018; 2017)). In particular, Hardt et al. (2016) shows gradient descent learns single-input-single-output (SISO) LDS with polynomial guarantees. Oymak & Ozay (2018) and Faradonbeh et al. (2018) provide sample complexity bounds for learning LDS. Sanandaji et al. (2011b;a) study the identification of sparse systems.

Neural networks: There is a growing literature on the theoretical aspects of deep learning and provable algorithms for training neural networks. Most of the existing results are concerned with feedforward networks. Ge et al. (2018); Li & Yuan (2017); Mei et al. (2018); Soltanolkotabi (2017); Janzamin et al. (2015); Soltanolkotabi et al. (2017); Zhong et al. (2017b) consider learning fully-connected shallow networks with gradient descent. Brutzkus & Globerson (2017); Zhong et al. (2017a); Du et al. (2017) address convolutional neural networks; which is an efficient weight-sharing architecture. Brutzkus et al. (2017); Wang et al. (2018) studies over-parameterized networks when data is linearly separable. Janzamin et al. (2015); Oymak & Soltanolkotabi (2018) utilize tensor decomposition techniques for learning feedforward neural nets. For recurrent networks, Sedghi & Anandkumar (2016) proposed tensor algorithms with polynomial guarantees and Khrulkov et al. (2017) studied their expressive power. More recently, Miller & Hardt (2018) showed that stable RNNs can be approximated by feed-forward networks.

# 2 PROBLEM SETUP

We first introduce the notation.  $\|\cdot\|$  returns the spectral norm of a matrix and  $s_{\min}(\cdot)$  returns the minimum singular value. The activation  $\phi: \mathbb{R} \to \mathbb{R}$  applies entry-wise if its input is a vector. Throughout,  $\phi$  is assumed to be a 1-Lipschitz function. With proper scaling of its parameters, the system (1.1) with a Lipschitz activation can be transformed into a system with 1-Lipschitz activation. The functions  $\pmb{\Sigma}[\cdot]$  and  $\mathbf{var}[\cdot]$  return the covariance of a random vector and variance of a random variable respectively.  $I_n$  is the identity matrix of size  $n \times n$ . Normal distribution with mean  $\pmb{\mu}$  and covariance  $\pmb{\Sigma}$  is denoted by  $\mathcal{N}(\pmb{\mu}, \pmb{\Sigma})$ . Throughout,  $c, C, c_0, c_1, \ldots$  denote positive absolute constants.

Setup: We consider the dynamical system parametrized by an activation function  $\phi(\cdot)$  and weight matrices  $A \in \mathbb{R}^{n \times n}$ ,  $B \in \mathbb{R}^{n \times p}$  as described in (1.1). Here,  $h_t$  is the  $n$  dimensional state-vector and  $u_t$  is the  $p$  dimensional input to the system at time  $t$ . As mentioned previously, (1.1) corresponds to the state equation of a recurrent neural network. For most RNNs of interest, the state  $h_t$  is hidden and we only get to interact with  $h_t$  via an additional output equation. For Elman networks Elman (1990), this equation is characterized by some output activation  $\phi_y$  and output weights  $C, D$  as follows

$$
\boldsymbol {y} _ {t} = \phi_ {y} \left(\boldsymbol {C h} _ {t} + \boldsymbol {D u} _ {t}\right). \tag {2.1}
$$

In this work, our attention is restricted to the state equation (1.1); which corresponds to setting  $\pmb{y}_t = \pmb{h}_t$  in the output equation. To analyze (1.1) in a non-asymptotic data-dependent setup, we assume a finite input/state trajectory of  $\{\pmb{u}_t, \pmb{h}_t\}_{t=0}^N$  generated by some ground truth weight matrices  $(\pmb{A}, \pmb{B})$ . Our goal is learning the unknown weights  $\pmb{A}$  and  $\pmb{B}$  in a data and computationally efficient

Algorithm 1 Learning state equations with nonlinear activations  
1: Inputs:  $(\pmb{y}_t, \pmb{h}_t, \pmb{u}_t)_{t=1}^N$  sampled from a trajectory. Scaling  $\mu$ , learning rate  $\eta$ . Initialization  $A_0, B_0$ .  
2: Outputs: Estimates  $\hat{\pmb{A}}, \hat{\pmb{B}}$  of the weight matrices  $\pmb{A}, \pmb{B}$ .  
3:  $\pmb{x}_t \gets [\mu \pmb{h}_t^T \pmb{u}_t^T]^T$  for  $1 \leq t \leq N$ .  
4:  $\Theta_0 \gets [\mu^{-1} A_0 B_0]$   
5: for  $\tau$  from 1 to END do  
6: Pick  $\gamma_\tau$  from  $\{1, 2, \dots, N\}$  uniformly at random.  
7:  $\Theta_\tau \gets \Theta_{\tau-1} - \eta \nabla \mathcal{L}_{\gamma_\tau}(\Theta_{\tau-1})$   
8: end for  
9: return  $[\hat{\pmb{A}}, \hat{\pmb{B}}] \gets \Theta_{\mathrm{END}} \begin{bmatrix} \mu I_n & 0 \\ 0 & I_p \end{bmatrix}$ .

way. In essence, we will show that, if the trajectory length satisfies  $N \gtrsim n + p$ , SGD can quickly and provably accomplish this goal using a constant step size.

Approach: Our approach is described in Algorithm 1. It takes two hyperparameters; the scaling factor  $\mu$  and learning rate  $\eta$ . Using the RNN trajectory, we construct  $N$  triples of the form  $\{\pmb{u}_t, \pmb{h}_t, \pmb{h}_{t+1}\}_{t=1}^N$ . We formulate a regression problem by defining the output vector  $\pmb{y}_t$ , input vector  $\pmb{x}_t$ , and the target parameter  $\pmb{C}$  as follows

$$
\boldsymbol {y} _ {t} = \boldsymbol {h} _ {t + 1}, \quad \boldsymbol {x} _ {t} = \left[ \begin{array}{c} \mu \boldsymbol {h} _ {t} \\ \boldsymbol {u} _ {t} \end{array} \right] \in \mathbb {R} ^ {n + p}, \quad \boldsymbol {C} = \left[ \mu^ {- 1} \boldsymbol {A} \boldsymbol {B} \right] \in \mathbb {R} ^ {n \times (n + p)}. \tag {2.2}
$$

With this reparameterization, we find the input/output identity  $\pmb{y}_t = \phi(\pmb{C}\pmb{x}_t)$ . We will consider the least-squares regression given by

$$
\mathcal {L} (\boldsymbol {\Theta}) = \frac {1}{N} \sum_ {t = 1} ^ {N} \mathcal {L} _ {t} (\boldsymbol {\Theta}) \quad \text {w h e r e} \quad \mathcal {L} _ {t} (\boldsymbol {\Theta}) = \frac {1}{2} \| \boldsymbol {y} _ {t} - \phi (\boldsymbol {\Theta} \boldsymbol {x} _ {t}) \| _ {\ell_ {2}} ^ {2}. \tag {2.3}
$$

For learning the ground truth parameter  $C$ , we utilize SGD on the loss function (2.3) with a constant learning rate  $\eta$ . Starting from an initial point  $\Theta_0$ , after END SGD iterations, Algorithm 1 returns an estimate  $\hat{C} = \Theta_{\mathrm{END}}$ . Estimates of  $A$  and  $B$  are decoded from the left and right submatrices of  $\hat{C}$  respectively.

# 3 MAIN RESULTS

# 3.1 PRELIMINARIES

The analysis of the state equation naturally depends on the choice of the activation function; which is the source of nonlinearity. We first define a class of Lipschitz and increasing activation functions.

Definition 3.1 ( $\beta$ -increasing activation). Given  $1 \geq \beta \geq 0$ , the activation function  $\phi$  satisfies  $\phi(0) = 0$  and  $1 \geq \phi'(x) \geq \beta$  for all  $x \in \mathbb{R}$ .

Our results will apply to strictly increasing activations where  $\phi$  is  $\beta$ -increasing for some  $\beta > 0$ . Observe that, this excludes ReLU activation which has zero derivative for negative values. However, it includes Leaky ReLU which is a generalization of ReLU. Parameterized by  $1 \geq \beta \geq 0$ , Leaky ReLU is a  $\beta$ -increasing function given by

$$
\operatorname {L R e L U} (x) = \max  (\beta x, x). \tag {3.1}
$$

In general, given an increasing and 1-Lipschitz activation  $\phi$ , a  $\beta$ -increasing function  $\phi_{\beta}$  can be obtained by blending  $\phi$  with the linear activation, i.e.  $\phi_{\beta}(x) = (1 - \beta)\phi(x) + \beta x$ .

A critical property that enables SGD is that the state-vector covariance  $\pmb{\Sigma}[\pmb{h}_t]$  is well-conditioned under proper assumptions. The lemma below provides upper and lower bounds on this covariance matrix in terms of problem variables.

Lemma 3.2 (State vector covariance). Consider the state equation (1.1) where  $\pmb{h}_0 = 0$  and  $\pmb{u}_t \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \pmb{I}_p)$ . Define the upper bound term  $B_t$  as

$$
B _ {t} = \| \boldsymbol {B} \| \sqrt {\frac {1 - \| \boldsymbol {A} \| ^ {2 t}}{1 - \| \boldsymbol {A} \| ^ {2}}}. \tag {3.2}
$$

- Suppose  $\phi$  is 1-Lipschitz and  $\phi(0) = 0$ . Then, for all  $t \geq 0$ ,  $\pmb{\Sigma}[\pmb{h}_t] \preceq B_t^2 \pmb{I}_n$ .  
Suppose  $\phi$  is a  $\beta$ -increasing function and  $p \geq n$ . Then,  $\Sigma[h_t] \succeq \beta^2 s_{\min}(\boldsymbol{B})^2 \boldsymbol{I}_n$ .

As a natural extension from linear dynamical systems, we will say the system is stable if  $\| \mathbf{A} \| < 1$  and unstable otherwise. For activations we consider, stability implies that if the input is set to 0, state vector  $\mathbf{h}_t$  will exponentially converge to 0 i.e. the system forgets the past states quickly. This is also the reason  $(B_t)_{t \geq 0}$  sequence converges for stable systems and diverges otherwise. The condition number of the covariance will play a critical role in our analysis. Using Lemma 3.2, this number can be upper bounded by  $\rho$  defined as

$$
\rho = \left(\frac {B _ {\infty}}{\beta s _ {\min } (\boldsymbol {B})}\right) ^ {2} = \left(\frac {\| \boldsymbol {B} \|}{s _ {\min } (\boldsymbol {B})}\right) ^ {2} \frac {1}{\beta^ {2} (1 - \| \boldsymbol {A} \| ^ {2})}. \tag {3.3}
$$

Observe that, the condition number of  $B$  appears inside the  $\rho$  term.

# 3.2 LEARNING FROM SINGLE TRAJECTORY

Our main result applies to stable systems  $(\| A \| < 1)$  and provides a non-asymptotic convergence guarantee for SGD in terms of the upper bound on the state vector covariance. This result characterizes the sample complexity and the rate of convergence of SGD; and also provides insights into the role of activation function and the spectral norm of  $A$ .

Theorem 3.3 (Main result). Let  $\{\pmb{u}_t, \pmb{h}_{t+1}\}_{t=1}^N$  be a finite trajectory generated from the state equation (1.1). Suppose  $\|\pmb{A}\| < 1$ ,  $\phi$  is  $\beta$ -increasing,  $\pmb{h}_0 = 0$ ,  $p \geq n$ , and  $\pmb{u}_t \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \pmb{I}_p)$ . Let  $\rho$  be same as (3.3) and  $c, C, c_0$  be properly chosen absolute constants. Pick the trajectory length  $N$  to satisfy

$$
N \geq C L \rho^ {2} (n + p),
$$

where  $L = 1 - \frac{\log(cn\rho)}{\log\|\mathbf{A}\|}$ . Pick scaling  $\mu = 1 / B_{\infty}$ , learning rate  $\eta = c_0\frac{\beta^2}{\rho n(n + p)}$ , and consider the loss function (2.3). With probability  $1 - 4N\exp (-100n) - 8L\exp (-\mathcal{O}(\frac{N}{L\rho^2}))$ , starting from an initial point  $\Theta_0$ , for all  $\tau \geq 0$ , the SGD iterations described in Algorithm 1 satisfies

$$
\mathbb {E} \left[ \| \boldsymbol {\Theta} _ {\tau} - \boldsymbol {C} \| _ {F} ^ {2} \right] \leq \left(1 - c _ {0} \frac {\beta^ {4}}{2 \rho^ {2} n (n + p)}\right) ^ {\tau} \| \boldsymbol {\Theta} _ {0} - \boldsymbol {C} \| _ {F} ^ {2}. \tag {3.4}
$$

Here the expectation is over the randomness of the SGD updates.

Sample complexity: Theorem 3.3 essentially requires  $N \gtrsim (n + p) / \beta^4$  samples for learning. This can be seen by unpacking (3.3) and ignoring the logarithmic  $L$  term and the condition number of  $B$ . Observe that  $\mathcal{O}(n + p)$  growth achieves near-optimal sample size for our problem. Each state equation (1.1) consists of  $n$  sub-equations (one for each entry of  $h_{t+1}$ ). We collect  $N$  state equations to obtain a system of  $Nn$  equations. On the other hand, the total number of unknown parameters in  $A$  and  $B$  are  $n(n + p)$ . This implies Theorem 3.3 is applicable as soon as the problem is mildly overdetermined i.e.  $Nn \gtrsim n(n + p)$ .

Computational complexity: Theorem 3.3 requires  $\mathcal{O}(n(n + p)\log \frac{1}{\varepsilon})$  iterations to reach  $\varepsilon$ -neighborhood of the ground truth. Our analysis reveals that, this rate can be accelerated if the state vector is zero-mean. This happens for odd activation functions satisfying  $\phi(-x) = -\phi(x)$  (e.g. linear activation). The result below is a corollary and requires  $\times n$  less iterations.

Theorem 3.4 (Faster learning for odd activations). Consider the same setup provided in Theorem 3.3. Additionally, assume that  $\phi$  is an odd function. Pick scaling  $\mu = 1 / B_{\infty}$ , learning rate  $\eta = c_0\frac{\beta^2}{\rho(n + p)}$ , and consider the loss function (2.3). With probability  $1 - 4N\exp (-100n) - 8L\exp (-\mathcal{O}(\frac{N}{L\rho^2}))$ , starting from an initial point  $\Theta_0$ , for all  $\tau \geq 0$ , the SGD iterations described in Algorithm 1 satisfies

$$
\mathbb {E} \left[ \| \boldsymbol {\Theta} _ {\tau} - \boldsymbol {C} \| _ {F} ^ {2} \right] \leq \left(1 - c _ {0} \frac {\beta^ {4}}{2 \rho^ {2} (n + p)}\right) ^ {\tau} \| \boldsymbol {\Theta} _ {0} - \boldsymbol {C} \| _ {F} ^ {2}, \tag {3.5}
$$

where the expectation is over the randomness of the SGD updates.

Another aspect of the convergence rate is the dependence on  $\beta$ . In terms of  $\beta$ , the SGD error (3.4) decays as  $(1 - \mathcal{O}(\beta^8))^{\tau}$ . While it is not clear how optimal is the exponent 8, numerical experiments in Section 6 demonstrate that larger  $\beta$  indeed results in drastically faster convergence.

# 4 MAIN IDEAS AND PROOF STRATEGY

To prove the results of the previous section, we derive a deterministic result that establishes the linear convergence of SGD for  $\beta$ -increasing functions. For linear convergence proofs, a typical strategy is showing the strong convexity of the loss function i.e. showing that, for some  $\alpha > 0$  and all points  $\pmb{v}, \pmb{u}$ , the gradient satisfies

$$
\langle \nabla \mathcal {L} (\boldsymbol {v}) - \nabla \mathcal {L} (\boldsymbol {u}), \boldsymbol {v} - \boldsymbol {u} \rangle \geq \alpha \| \boldsymbol {v} - \boldsymbol {u} \| _ {\ell_ {2}} ^ {2}.
$$

The core idea of our convergence result is that the strong convexity parameter of the loss function with  $\beta$ -increasing activations can be connected to the loss function with linear activations. In particular, recalling (2.3), set  $\pmb{y}_t^{\mathrm{lin}} = \pmb{C}\pmb{x}_t$  and define the linear loss to be

$$
\mathcal {L} ^ {\mathrm {l i n}} (\boldsymbol {\Theta}) = \frac {1}{2 N} \sum_ {i = 1} ^ {N} \| \boldsymbol {y} _ {t} ^ {\mathrm {l i n}} - \boldsymbol {\Theta} \boldsymbol {x} _ {t} \| _ {\ell_ {2}} ^ {2}.
$$

Denoting the strong convexity parameter of the original loss by  $\alpha_{\phi}$  and that of linear loss by  $\alpha_{\mathrm{lin}}$ , we argue that  $\alpha_{\phi} \geq \beta^{2}\alpha_{\mathrm{lin}}$ ; which allows us to establish a convergence result as soon as  $\alpha_{\mathrm{lin}}$  is strictly positive. Next result is our SGD convergence theorem which follows from this discussion.

Theorem 4.1 (Deterministic convergence). Suppose a data set  $\{\pmb{x}_i, \pmb{y}_i\}_{i=1}^N$  is given; where output  $\pmb{y}_i$  is related to input  $\pmb{x}_i$  via  $\pmb{y}_i = \phi(\langle \pmb{x}_i, \pmb{\theta} \rangle)$  for some  $\pmb{\theta} \in \mathbb{R}^n$ . Suppose  $\beta > 0$  and  $\phi$  is a  $\beta$ -increasing. Let  $\gamma_+ \geq \gamma_- > 0$  be scalars. Assume that input samples satisfy the bounds

$$
\gamma_ {+} \boldsymbol {I} _ {n} \succeq \frac {1}{N} \sum_ {i = 1} ^ {N} \boldsymbol {x} _ {i} \boldsymbol {x} _ {i} ^ {T} \succeq \gamma_ {-} \boldsymbol {I} _ {n}, \| \boldsymbol {x} _ {i} \| _ {\ell_ {2}} ^ {2} \leq B f o r a l l i.
$$

Let  $\{r_{\tau}\}_{\tau = 0}^{\infty}$  be a sequence of i.i.d. integers uniformly distributed between 1 to  $N$ . Then, starting from an arbitrary point  $\theta_0$ , setting learning rate  $\eta = \frac{\beta^2\gamma_-}{\gamma + B}$ , for all  $\tau \geq 0$ , the SGD iterations for quadratic loss

$$
\boldsymbol {\theta} _ {\tau + 1} = \boldsymbol {\theta} _ {\tau} - \eta \left(\phi \left(\boldsymbol {x} _ {r _ {\tau}} ^ {T} \boldsymbol {\theta} _ {\tau}\right) - \boldsymbol {y} _ {r _ {\tau}}\right) \phi^ {\prime} \left(\boldsymbol {x} _ {r _ {\tau}} ^ {T} \boldsymbol {\theta} _ {\tau}\right) \boldsymbol {x} _ {r _ {\tau}}, \tag {4.1}
$$

satisfies the error bound

$$
\mathbb {E} \left[ \| \boldsymbol {\theta} _ {\tau} - \boldsymbol {\theta} \| _ {\ell_ {2}} ^ {2} \right] \leq \| \boldsymbol {\theta} _ {0} - \boldsymbol {\theta} \| _ {\ell_ {2}} ^ {2} \left(1 - \frac {\beta^ {4} \gamma_ {-} ^ {2}}{\gamma_ {+} B}\right) ^ {\tau}, \tag {4.2}
$$

where the expectation is over the random selection of the SGD iterations  $\{r_{\tau}\}_{\tau = 0}^{\infty}$ .

This theorem provides a clean convergence rate for SGD for  $\beta$ -increasing activations and naturally generalizes standard results on linear regression which corresponds to  $\beta = 1$ . Its extension to proximal gradient methods might be beneficial for high-dimensional nonlinear problems (e.g. sparse/low-rank approximation and generalized linear models Cai et al. (2010); Beck & Teboulle (2009); Oymak et al. (2018); Agarwal et al. (2010)) and is left as a future work.

To derive the results from Section 3, we need to determine the conditions under which Theorem 4.1 is applicable to the data obtained from RNN state equation with high probability. Below we provide desirable characteristics of the state vector; which enables our statistical results.

Assumption 1 (Well-behaved state vector). Let  $L > 1$  be an integer. There exists positive scalars  $\gamma_{+}, \gamma_{-}, \theta$  and an absolute constant  $C > 0$  such that  $\theta \leq 3\sqrt{n}$  and the following holds

- Lower bound:  $\Sigma[h_{L - 1}] \succeq \gamma_{-} I_n$ ,  
- Upper bound: for all  $t$ , the state vector satisfies

$$
\boldsymbol {\Sigma} \left[ \boldsymbol {h} _ {t} \right] \preceq \gamma_ {+} \boldsymbol {I} _ {n}, \| \boldsymbol {h} _ {t} - \mathbb {E} [ \boldsymbol {h} _ {t} ] \| _ {\psi_ {2}} \leq C \sqrt {\gamma_ {+}} \quad a n d \quad \| \mathbb {E} [ \boldsymbol {h} _ {t} ] \| _ {\ell_ {2}} \leq \theta \sqrt {\gamma_ {+}}. \tag {4.3}
$$

Here  $\| \cdot \|_{\psi_2}$  returns the subgaussian norm of a vector (see Def. 5.22 of Vershynin (2010)).

Assumption 1 ensures that covariance is well-conditioned, state vector is well-concentrated, and it has a reasonably small expectation. Our next theorem establishes statistical guarantees for learning the RNN state equation based on this assumption.

Theorem 4.2 (General result). Let  $\{\pmb{u}_t, \pmb{h}_{t+1}\}_{t=1}^N$  be a length  $N$  trajectory of the state equation (1.1). Suppose  $\| \pmb{A} \| < 1$ ,  $\phi$  is  $\beta$ -increasing,  $\pmb{h}_0 = 0$ , and  $\pmb{u}_t \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \pmb{I}_p)$ . Given scalars  $\gamma_+ \geq \gamma_- > 0$ , set the condition number as  $\rho = \gamma_+ / \gamma_-$ . For absolute constants  $C, c, c_0 > 0$ , choose trajectory length  $N$  to satisfy

$$
N \geq C L \rho^ {2} (n + p) \quad w h e r e \quad L = \lceil 1 - \frac {\log (c n \rho)}{\log \| \boldsymbol {A} \|} \rceil .
$$

Suppose Assumption 1 holds with  $L, \gamma_{+}, \gamma_{-}, \theta$ . Pick scaling to be  $\mu = 1 / \sqrt{\gamma_{+}}$  and learning rate to be  $\eta = c_0 \frac{\beta^2}{\rho(\theta + \sqrt{2})^2(n + p)}$ . With probability  $1 - 4N \exp(-100n) - 8L \exp(-\mathcal{O}(\frac{N}{L\rho^2}))$ , starting from  $\Theta_0$ , for all  $\tau \geq 0$ , the SGD iterations on loss (2.3) as described in Algorithm 1 satisfies

$$
\mathbb {E} \left[ \| \boldsymbol {\Theta} _ {\tau} - \boldsymbol {C} \| _ {F} ^ {2} \right] \leq \left(1 - c _ {0} \frac {\beta^ {4}}{2 \rho^ {2} (\theta + \sqrt {2}) ^ {2} (n + p)}\right) ^ {\tau} \| \boldsymbol {\Theta} _ {0} - \boldsymbol {C} \| _ {F} ^ {2}, \tag {4.4}
$$

where the expectation is over the randomness of SGD updates.

The advantage of this theorem is that, it isolates the optimization problem from the statistical properties of state vector. If one can prove tighter bounds on achievable  $(\gamma_{+},\gamma_{-},\theta)$ , it will immediately imply improved performance for SGD. In particular, Theorems 3.3 and 3.4 are simple corollaries of Theorem 4.2 with proper choices.

- Theorem 3.3 follows by setting  $\gamma_{+} = B_{\infty}^{2}$ ,  $\gamma_{-} = \beta^{2}s_{\min}(B)^{2}$ , and  $\theta = \sqrt{n}$ .  
- Theorem 3.4 follows by setting  $\gamma_{+} = B_{\infty}^{2}$ ,  $\gamma_{-} = \beta^{2}s_{\min}(B)^{2}$ , and  $\theta = 0$ .

# 5 LEARNING UNSTABLE SYSTEMS

So far, we considered learning from a single RNN trajectory for stable systems  $(\| A \| < 1)$ . For such systems, as the time goes on, the impact of the earlier states disappear. In our analysis, this allows us to split a single trajectory into multiple nearly-independent trajectories. This approach will not work for unstable systems ( $A$  is arbitrary) where the impact of older states may be amplified over time. To address this, we consider a model where the data is sampled from multiple independent trajectories.

Suppose  $N$  independent trajectories of the state-equation (1.1) are available. Pick some integer  $T_0 \geq 1$ . Denoting the  $i$ th trajectory by the triple  $(\pmb{h}_{t+1}^{(i)}, \pmb{h}_t^{(i)}, \pmb{u}_t^{(i)})_{t \geq 0}$ , we collect a single sample from each trajectory at time  $T_0$  to obtain the triple  $(\pmb{h}_{T_0+1}^{(i)}, \pmb{h}_{T_0}^{(i)}, \pmb{u}_{T_0}^{(i)})$ . To utilize the existing optimization framework (2.3); for  $1 \leq i \leq N$ , we set,

$$
\left(\boldsymbol {y} _ {i}, \boldsymbol {h} _ {i}, \boldsymbol {u} _ {i}\right) = \left(\boldsymbol {h} _ {T _ {0} + 1} ^ {(i)}, \boldsymbol {h} _ {T _ {0}} ^ {(i)}, \boldsymbol {u} _ {T _ {0}} ^ {(i)}\right). \tag {5.1}
$$

With this setup, we can again use the SGD Algorithm 1 to learn the weights  $\mathbf{A}$  and  $\mathbf{B}$ . The crucial difference compared to Section 3 is that, the samples  $(\pmb{y}_i, \pmb{h}_i, \pmb{u}_i)_{i=1}^N$  are now independent of each other; hence, the analysis is simplified. As previously, having an upper bound on the condition number of the state-vector covariance is critical. This upper bound can be shown to be  $\rho$  defined as

$$
\rho = \left\{ \begin{array}{l} \bar {\rho} \quad \text {i f} n > 1 \\ \bar {\rho} \frac {1 - \beta^ {2} | \boldsymbol {A} | ^ {2}}{1 - (\beta | \boldsymbol {A} |) ^ {2 T _ {0}}} \quad \text {i f} n = 1 \end{array} \right. \quad \text {w h e r e} \quad \bar {\rho} = \frac {B _ {T _ {0}} ^ {2}}{\beta^ {2} s _ {\min } (\boldsymbol {B}) ^ {2}}. \tag {5.2}
$$

The  $\bar{\rho}$  term is similar to the earlier definition (3.3); however it involves  $B_{T_0}$  rather than  $B_{\infty}$ . This modification is indeed necessary since  $B_{\infty} = \infty$  when  $\| \mathbf{A}\| > 1$ . On the other hand, note that,  $B_{T_0}^2$  grows proportional to  $\| \mathbf{A}\|^{2T_0}$ ; which results in exponentially bad condition number in  $T_0$ . Our  $\rho$  definition remedies this issue for single-output systems; where  $n = 1$  and  $\mathbf{A}$  is a scalar. In particular, when  $\beta = 1$  (e.g.  $\phi$  is linear)  $\rho$  becomes equal to the correct value  $1^2$ . The next theorem provides our result on unstable systems in terms of this condition number and other model parameters.

# Algorithm 2 Empirical hyperparameter selection.

1: Inputs:  $(\pmb{h}_t, \pmb{u}_t)_{t=1}^N$  sampled from a trajectory.  
2: Outputs: Scaling  $\mu$ .  
3: Form the empirical covariance matrix  $\Sigma_h$  from  $\{\pmb{h}_t\}_{t = 1}^N$ .  
4: Form the empirical covariance matrix  $\Sigma_{u}$  from  $\{\pmb{u}_t\}_{t=1}^N$ .  
5: return  $\sqrt{\|\Sigma_u\| / \|\Sigma_h\|}$ .

![](images/7ac81c9a600ef9c04c4b145bb8127e704ef10d720905a62a6796aaaa48f10966.jpg)  
(a)

![](images/92577cb6b9d05eca3109ad2d3cd9750e249474a092d2f04cebdc8b5b140cee7e.jpg)  
(b)  
Figure 1: SGD convergence behavior for Leaky RLUs with varying minimum slope  $\beta$ . Figures a) and b) repeat the same experiments. The difference is the spectral norm of the ground truth state matrix  $A$ .

Theorem 5.1 (Unstable systems). Suppose we are given  $N$  independent trajectories  $(\pmb{h}_t^{(i)},\pmb{u}_t^{(i)})_{t\geq 0}$  for  $1\leq i\leq N$ . Each trajectory is sampled at time  $T_0$  to obtain  $N$  samples  $(\pmb{y}_i,\pmb {h}_i,\pmb {u}_i)^{N}_{i = 1}$  where the ith sample is given by (5.1). Suppose the sample size satisfies

$$
N \geq C \rho^ {2} (n + p)
$$

where  $\rho$  is given by (5.2). Assume the initial states are  $0$ ,  $\phi$  is  $\beta$ -increasing,  $p \geq n$ , and  $\mathbf{u}_t \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \mathbf{I}_p)$ . Set scaling  $\mu = 1 / \sqrt{B_{T_0}}$ , learning rate  $\eta = c_0 \frac{\beta^2}{\rho n(n + p)}$ , and run SGD over the equations described in (2.2) and (2.3). Starting from  $\Theta_0$ , with probability  $1 - 2N \exp(-100(n + p)) - 4 \exp(-\mathcal{O}\left(\frac{N}{\rho^2}\right))$ , all SGD iterations satisfy

$$
\mathbb {E} [ \| \boldsymbol {\Theta} _ {\tau} - \boldsymbol {C} \| _ {F} ^ {2} ] \leq (1 - c _ {0} \frac {\beta^ {4}}{2 \rho^ {2} n (n + p)}) ^ {\tau} \| \boldsymbol {\Theta} _ {0} - \boldsymbol {C} \| _ {F} ^ {2},
$$

where the expectation is over the randomness of the SGD updates.

# 6 NUMERICAL EXPERIMENTS

We conducted experiments on ReLU and Leaky ReLU activations. Let us first describe the experimental setup. We pick the state dimension  $n = 50$  and the input dimension  $p = 100$ . We choose the ground truth matrix  $A$  to be a scaled random unitary matrix; which ensures that all singular values of  $A$  are equal.  $B$  is generated with i.i.d.  $\mathcal{N}(0,1)$  entries. Instead of using the theoretical scaling choice, we determine the scaling  $\mu$  from empirical covariance matrices outlined in Algorithm 2. Similar to our proof strategy, this algorithm equalizes the spectral norms of the input and state covariances to speed up convergence. We also empirically determined the learning rate and used  $\eta = 1 / 100$  in all experiments.

Evaluation: We consider two performance measures in the experiments. Let  $\hat{C}$  be an estimate of the ground truth parameter  $C = [\mu^{-1}AB]$ . The first measure is the normalized error defined as  $\| \hat{C} - C\|_F^2 / \| C\|_F^2$ . The second measure is the normalized loss defined as

$$
\frac {\sum_ {i = 1} ^ {N} \left\| \boldsymbol {y} _ {t} - \phi (\hat {\boldsymbol {C}} \boldsymbol {x} _ {t}) \right\| _ {\ell_ {2}} ^ {2}}{\sum_ {i = 1} ^ {N} \left\| \boldsymbol {y} _ {t} \right\| _ {\ell_ {2}} ^ {2}}.
$$

In all experiments, we run Algorithm 1 for 50000 SGD iterations and plot these measures as a function of  $\tau$ ; by using the estimate available at the end of the  $\tau$ th SGD iteration for  $0 \leq \tau \leq 50000$ . Each curve is obtained by averaging the outcomes of 20 independent realizations.

![](images/834a8ad68e83b18ee8f5c75d765e667aa0f64e373cadc406ca5cde9543cd188c.jpg)  
(a)

![](images/dee5425a1cf93caac62243e8d381398a38d658e09535bb77f65479ebfbfe2778.jpg)  
(b)  
Figure 2: SGD convergence behavior for ReLU with varying spectral norm of the state matrix  $\mathbf{A}$ . Figures a) and b) repeats the same experiments. The difference is that a) uses  $N = 500$  trajectory length whereas b) uses  $N = 2500$  (i.e.  $\times 5$  more data). Shaded regions highlight the one standard deviation around the mean.

Our first experiments use  $N = 500$ ; which is mildly larger than the total dimension  $n + p = 150$ . In Figure 1, we plot the Leaky ReLU errors with varying slopes as described in (3.1). Here  $\beta = 0$  corresponds to ReLU and  $\beta = 1$  is the linear model. In consistence with our theory, SGD achieves linear convergence and as  $\beta$  increases, the rate of convergence drastically improves. The improvement is more visible for less stable systems driven by  $A$  with a larger spectral norm. In particular, while ReLU converges for small  $\| A \|$ , SGD gets stuck before reaching the ground truth when  $\| A \| = 0.8$ .

To understand, how well SGD fits the training data, in Figure 2a, we plotted the normalized loss for ReLU activation. For more unstable system  $(\| A \| = 0.9)$ , training loss stagnates in a similar fashion to the parameter error. We also verified that the norm of the overall gradient  $\|\nabla \mathcal{L}(\Theta_{\tau})\|_F$  continues to decay (where  $\Theta_{\tau}$  is the  $\tau$ th SGD iterate); which implies that SGD converges before reaching a global minima. As  $A$  becomes more stable, rate of convergence improves and linear rate is visible. Finally, to better understand the population landscape of the quadratic loss with ReLU activations, Figure 2b repeats the same ReLU experiments while increasing the sample size five times to  $N = 2500$ . For this more overdetermined problem, SGD converges even for  $\| A \| = 0.9$ ; indicating that

- population landscape of loss with ReLU activation is well-behaved,  
- however ReLU problem requires more data compared to the Leaky ReLU for finding global minima.

Overall, as predicted by our theory, experiments verify that SGD indeed quickly finds the optimal weight matrices of the state equation (1.1) and as the activation slope  $\beta$  increases, the convergence rate improves.

# 7 CONCLUSIONS

This work showed that SGD can learn the nonlinear dynamical system (1.1); which is characterized by weight matrices and an activation function. This problem is of interest for recurrent neural networks as well as nonlinear system identification. We showed that efficient learning is possible with optimal sample complexity and good computational performance. Our results apply to strictly increasing activations such as Leaky ReLU. We empirically showed that Leaky ReLU converges faster than ReLU and requires less samples; in consistence with our theory. We list a few unanswered problems that would provide further insights into recurrent neural networks.

- Covariance of the state-vector: Our results depend on the covariance of the state-vector and requires it to be positive definite. One might be able to improve the current bounds on the condition number and relax the assumptions on the activation function. Deriving similar performance bounds for ReLU is particularly interesting.  
- Hidden state: For RNNs, the state vector is hidden and is observed through an additional equation (2.1); which further complicates the optimization landscape. Even for linear dynamical systems, learning the  $(A,B,C,D)$  system ((1.1), (2.1)) is a non-trivial task Ho & Kalman (1966); Hardt et al. (2016). What can be said when we add the nonlinear activations?  
- Classification task: In this work, we used normally distributed input and least-squares regression for our theoretical guarantees. More realistic input distributions might provide better insight into contemporary problems, such as natural language processing; where the goal is closer to classification (e.g. finding the best translation from another language).

# REFERENCES

Alekh Agarwal, Sahand Negahban, and Martin J Wainwright. Fast global convergence rates of gradient methods for high-dimensional statistical recovery. In Advances in Neural Information Processing Systems, pp. 37-45, 2010.  
Karl Johan Åström and Peter Eykhoff. System identification—a survey. Automatica, 7(2):123-162, 1971.  
Karl Johan Åström and Tore Hägglund. PID controllers: theory, design, and tuning, volume 2. Instrument society of America Research Triangle Park, NC, 1995.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1):183-202, 2009.  
Robert Grover Brown, Patrick YC Hwang, et al. Introduction to random signals and applied Kalman filtering, volume 3. Wiley New York, 1992.  
Alon Brutzkus and Amir Globerson. Globally optimal gradient descent for a convnet with gaussian inputs. arXiv preprint arXiv:1702.07966, 2017.  
Alon Brutzkus, Amir Globerson, Eran Malach, and Shai Shalev-Shwartz. Sgd learns over-parameterized networks that provably generalize on linearly separable data. arXiv preprint arXiv:1710.10174, 2017.  
Jian-Feng Cai, Emmanuel J Candès, and Zuowei Shen. A singular value thresholding algorithm for matrix completion. SIAM Journal on Optimization, 20(4):1956-1982, 2010.  
S. Dirksen. Tail bounds via generic chaining. arXiv preprint arXiv:1309.3522, 2013.  
Simon S Du, Jason D Lee, and Yuandong Tian. When is a convolutional filter easy to learn? arXiv preprint arXiv:1709.06129, 2017.  
Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179-211, 1990.  
Mohamad Kazem Shirani Faradonbeh, Ambuj Tewari, and George Michailidis. Finite time identification in unstable linear systems. Automatica, 96:342-353, 2018.  
Rong Ge, Jason D Lee, and Tengyu Ma. Learning one-hidden-layer neural networks with landscape design. *ICLR*, 2018.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In Acoustics, speech and signal processing (icassp), 2013 IEEE international conference on, pp. 6645-6649. IEEE, 2013.  
Moritz Hardt, Tengyu Ma, and Benjamin Recht. Gradient descent learns linear dynamical systems. arXiv preprint arXiv:1609.05191, 2016.  
BL Ho and Rudolph E Kalman. Effective construction of linear state-variable models from input/output functions. at-Automatisierungstechnik, 14(1-12):545-548, 1966.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.  
Majid Janzamin, Hanie Sedghi, and Anima Anandkumar. Beating the perils of non-convexity: Guaranteed training of neural networks using tensor methods. arXiv preprint arXiv:1506.08473, 2015.  
Valentin Khrulkov, Alexander Novikov, and Ivan Oseledets. Expressive power of recurrent neural networks. arXiv preprint arXiv:1711.00811, 2017.  
Michel Ledoux. The concentration of measure phenomenon. American Mathematical Soc., 2001.  
Yuanzhi Li and Yang Yuan. Convergence analysis of two-layer neural networks with relu activation. In Advances in Neural Information Processing Systems, pp. 597-607, 2017.  
Lennart Ljung. System identification: theory for the user. Prentice-hall, 1987.  
Lennart Ljung. System identification. In Signal analysis and prediction, pp. 163-173. Springer, 1998.

Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layers neural networks. arXiv preprint arXiv:1804.06561, 2018.  
John Miller and Moritz Hardt. When recurrent models don't need to be recurrent. arXiv preprint arXiv:1805.10369, 2018.  
Samet Oymak and Necmiye Ozay. Non-asymptotic identification of lti systems from a single trajectory. arXiv preprint arXiv:1806.05722, 2018.  
Samet Oymak and Mahdi Soltanolkotabi. End-to-end learning of a convolutional neural network via deep tensor decomposition. arXiv preprint arXiv:1805.06523, 2018.  
Samet Oymak, Benjamin Recht, and Mahdi Soltanolkotabi. Sharp time-data tradeoffs for linear inverse problems. IEEE Transactions on Information Theory, 64(6):4129-4158, 2018.  
Borhan M Sanandaji, Tyrone L Vincent, and Michael B Wakin. Exact topology identification of large-scale interconnected dynamical systems from compressive observations. In American Control Conference (ACC), 2011, pp. 649-656. IEEE, 2011a.  
Borhan M Sanandaji, Tyrone L Vincent, Michael B Wakin, Roland Tóth, and Kameshwar Poolla. Compressive system identification of lti and ltv arx models. In Decision and Control and European Control Conference (CDC-ECC), 2011 50th IEEE Conference on, pp. 791-798. IEEE, 2011b.  
Hanie Sedghi and Anima Anandkumar. Training input-output recurrent neural networks through spectral methods. arXiv preprint arXiv:1603.00954, 2016.  
Max Simchowitz, Horia Mania, Stephen Tu, Michael I Jordan, and Benjamin Recht. Learning without mixing: Towards a sharp analysis of linear system identification. arXiv preprint arXiv:1802.08334, 2018.  
Mahdi Soltanolkotabi. Learning relus via gradient descent. arXiv preprint arXiv:1705.04591, 2017.  
Mahdi Soltanolkotabi, Adel Javanmard, and Jason D Lee. Theoretical insights into the optimization landscape of over-parameterized shallow neural networks. arXiv preprint arXiv:1707.04926, 2017.  
Michel Talagrand. Gaussian processes and the generic chaining. In Upper and Lower Bounds for Stochastic Processes, pp. 13-73. Springer, 2014.  
Stephen Tu, Ross Boczar, Andrew Packard, and Benjamin Recht. Non-asymptotic analysis of robust control from coarse-grained identification. arXiv preprint arXiv:1707.04791, 2017.  
Stephen Tu, Ross Boczar, and Benjamin Recht. On the approximation of toeplitz operators for nonparametric  $\langle_{\infty}$ -norm estimation. In 2018 Annual American Control Conference (ACC), pp. 1867-1872. IEEE, 2018.  
Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. arXiv preprint arXiv:1011.3027, 2010.  
Gang Wang, Georgios B Giannakis, and Jie Chen. Learning relu networks on linearly separable data: Algorithm, optimality, and generalization. arXiv preprint arXiv:1808.04685, 2018.  
Kai Zhong, Zhao Song, and Inderjit S Dhillion. Learning non-overlapping convolutional neural networks with multiple kernels. arXiv preprint arXiv:1711.03440, 2017a.  
Kai Zhong, Zhao Song, Prateek Jain, Peter L Bartlett, and Inderjit S Dhillon. Recovery guarantees for one-hidden-layer neural networks. arXiv preprint arXiv:1706.03175, 2017b.
