# SYNC: SAFETY-AWARE NEURAL CONTROL FOR STABILIZING STOCHASTIC DELAY-DIFFERENTIAL EQUATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Stabilization of the systems described by stochastic delay-differential equations is a challenging task in control community. Here, to achieve this task, we leverage neural networks to learn control policies using the information of the controlled systems in some prescribed regions. The two learned control policies, the neural deterministic controller (NDC) and the neural stochastic controller (NSC), work effectively because the learning procedures use, respectively, the well-known LaSalle-Type theorem and the newly-established theorem for guaranteeing the stochastic stability in SDDEs. We theoretically investigate the performance of the proposed NDC and NSC in terms of convergence time and energy cost. More practically and significantly, we improve our learned control policies through considering the situation where the controlled trajectories can only evolve in some specific safety set. Such successful stabilization based on neural networks restricted in safety set is attributed to our further developed theory for safety verification of SDDEs using the stochastic control barrier function, and we name it as SYNC (SafetY-aware Neural Control). The efficacy of all the articulated control policies, including the SYNC, is demonstrated systematically by using representative control problems.

# 1 INTRODUCTION

Stochastic delay-differential equations (SDDEs) (Mao, 1996; Lin & He, 2005; Sun & Cao, 2007; Guo et al., 2016) have been leveraged to characterize the complex dynamical behavior emergent in real-world systems with dependence on the current state, the past state, and the noise. Efficiently controlling these systems is a longstanding crucial problem, with the consequent emphasis being placed on the design of control policies and analysis of stability in SDDEs. Traditional control methods in stochastic settings have been fully developed in the convex optimization frameworks using the control Lyapunov stability theory, e.g. the quadratic programming (QP) (Fan et al., 2020; Sarkar et al., 2020). These methods cannot provide the analytical form of feedback controllers and own a high computational cost, requiring solving QP problems at each iteration step. To overcome these difficulties, utilizing neural networks (NNs) to

automatically design controllers becomes one of the mainstream approaches in recent years (Zhang et al., 2022; Chang et al., 2019). However, existing machine-learning-based methods either focus on controlling systems without time-delay or aim at learning the control Lyapunov function instead of the control policy (Khansari-Zadeh & Billard, 2014), promoting us to design neural controllers for general nonlinear SDDEs.

![](images/9923071929e9d9420473c54d9b7a40b6c702ab335e72eb51137d0d0f1640ef6d.jpg)  
Figure 1: Sketches of SYNC. Both the NDC and NSC can stabilize the SDDEs to the target unstable equilibrium  $x^{*}$ . The safety-aware controlled state trajectories are restricted in the safe region.

The safety verification of controlled systems plays an important role in many branches of cybernetics and industry. For example, with the safety verification, one can reduce a significant economic burden and loss of life (Ames et al., 2016; Wang et al., 2016). In particular, the dominant framework for safety control in stochastic settings is the use of stochastic control barrier function (SCBF) (Clark, 2019; Santoyo et al., 2021). The core idea of designing a candidate SCBF is that its value tends to explode as the system's state leaves the safe region, implying a safety guarantee as long as one could design a controller such that the SCBF is always finite within the controlled time duration. Unfortunately, the existing theories of SCBF either require a lot of inequality constraints or are limited in handling systems without time-delay.

In this paper, we utilize neural networks (NNs) to learn control policies for SDDEs based on the corresponding stability theories. Additionally, we develop a simplified SCBF theory for SDDEs and then use it to construct the neural controller with safety guarantee, named SYNC. All these control policies are intuitively depicted in Figure 1. The major contributions of this paper are listed as follows

- designing a novel and practical framework of neural deterministic control based on the existing LaSalle-Type stability theory,  
- proposing a simplified stability theorem and designing the second novel neural stochastic control framework that can benefit from noise according to this theorem,  
- providing theoretical estimation for proposed neural controller in terms of convergence time and energy cost,  
- cultivating a simplified SCBF theory for SDDEs and exploiting it in our neural framework to obtain a safety guarantee, and  
- demonstrating the efficacy of the proposed neural control methods through numerical comparisons with the typical existing control methods on several representative physical systems.

# 2 PRELIMINARIES

To begin with, we consider the SDDE of the following general form:

$$
\mathrm {d} \boldsymbol {x} (t) = F (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau), t) \mathrm {d} t + G (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau), t) \mathrm {d} B _ {t}, t \geq 0, \tau > 0, \boldsymbol {x} (t) \in \mathbb {R} ^ {d}, \tag {1}
$$

where  $\pmb{x}(t) = \xi(t) \in C_{\mathcal{F}_0}([- \tau, 0]; \mathbb{R}^d)$  is the initial function, the drift term  $F: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}_+ \to \mathbb{R}^d$  and the diffusion term  $G: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}_+ \to \mathbb{R}^{d \times r}$  are Borel-measurable functions, and  $B_t$  is a standard  $r$ -dimensional ( $r$ -D) Brownian motion defined on probability space  $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t \geq 0}, \mathbb{P})$  with a filtration  $\{\mathcal{F}_t\}_{t \geq 0}$  satisfying the regular conditions. Without loss of generality, we assume  $F(\mathbf{0}, \mathbf{0}, t) = \mathbf{0}$ ,  $G(\mathbf{0}, \mathbf{0}, t) = \mathbf{0}$  to guarantee that  $\pmb{x}(t) \equiv \mathbf{0}$ ,  $t \geq 0$  is an equilibrium of Eq. (1). For simplicity, the following notations and assumptions are used throughout the paper.

Assumption 2.1 Eq. (1) has a unique solution  $\pmb{x}(t,\xi)$  on  $t \geq 0$  for any  $\xi \in C_{\mathcal{F}_0}([- \tau, 0]; \mathbb{R}^d)$  and for every integer  $n \geq 1$ , there is a number  $K_n > 0$  such that

$$
\| F (\boldsymbol {x}, \boldsymbol {y}, t) \| \vee \| G (\boldsymbol {x}, \boldsymbol {y}, t) \| _ {\mathrm {F}} \leq K _ {n},
$$

for any  $(\pmb{x},\pmb{y},t)\in \mathbb{R}^d\times \mathbb{R}^d\times \mathbb{R}_+$  with  $\| \pmb {x}\| \vee \| \pmb {y}\| \leq n$  where  $\| \cdot \|$  denotes the  $L^2$  -norm and  $\| \cdot \|_{\mathrm{F}}$  denotes the Frobenius norm, i.e.  $\| G(\pmb {x},\pmb {y},t)\|_{\mathrm{F}}^{2} = \sum_{i = 1}^{d}\sum_{j = 1}^{r}G_{ij}(\pmb {x},\pmb {y},t)^{2}$

Definition 2.1 (Derivative Operator) Define the differential operator  $\mathcal{L}$  associated with Eq. (1) by

$$
\mathcal {L} \triangleq \frac {\partial}{\partial t} + \sum_ {i = 1} ^ {d} F _ {i} (\boldsymbol {x}, \boldsymbol {y}, t) \frac {\partial}{\partial x _ {i}} + \frac {1}{2} \sum_ {i, j = 1} ^ {d} \left[ G (\boldsymbol {x}, \boldsymbol {y}, t) G ^ {\top} (\boldsymbol {x}, \boldsymbol {y}, t) \right] _ {i j} \frac {\partial^ {2}}{\partial x _ {i} \partial x _ {j}}.
$$

According to the above definition of the derivative operator, an operation of  $\mathcal{L}$  on the function  $V\in C^{2,1}(\mathbb{R}^d\times \mathbb{R}_+;\mathbb{R})$  yields:

$$
\mathcal {L} V (\boldsymbol {x}, \boldsymbol {y}, t) = V _ {t} (\boldsymbol {x}, t) + \nabla V (\boldsymbol {x}, t) ^ {\top} F (\boldsymbol {x}, \boldsymbol {y}, t) + \frac {1}{2} \operatorname {T r} \left[ G ^ {\top} (\boldsymbol {x}, \boldsymbol {y}, t) \mathcal {H} V (\boldsymbol {x}, t) G (\boldsymbol {x}, \boldsymbol {y}, t) \right]. \tag {2}
$$

Here,  $V_{t}$ ,  $\nabla V$  and  $\mathcal{H}V$  represent, respectively, the time derivative, the gradient and the Hessian matrix of  $V$ . Notably, the following LaSalle-Type stability theorem will be used in the establishment of our part of the results.

Theorem 2.2 (Mao, 2002) Suppose that Assumptions 2.1 hold. Assume there are functions  $V \in C^{2,1}(\mathbb{R}^d \times \mathbb{R}_+; \mathbb{R}_+)$ ,  $\gamma \in L^1(\mathbb{R}_+; \mathbb{R}_+)$ , and  $w_1, w_2 \in C(\mathbb{R}^d; \mathbb{R}_+)$  such that

$$
\mathcal {L} V (\boldsymbol {x}, \boldsymbol {y}, t) \leq \gamma (t) - w _ {1} (\boldsymbol {x}) + w _ {2} (\boldsymbol {y}), w _ {1} (\boldsymbol {x}) \geq w _ {2} (\boldsymbol {x}), a n d \lim  _ {\| \boldsymbol {x} \| \rightarrow \infty} \inf  _ {0 \leq t \leq \infty} V (\boldsymbol {x}, t) = \infty .
$$

Then,  $\mathrm{Ker}(w_1 - w_2) \neq \varnothing$  and  $\lim_{t \to \infty} \mathrm{dist}(\pmb{x}(t,\xi), \mathrm{Ker}(w_1 - w_2)) = 0$  a.s., where  $\mathrm{Ker}(w_1 - w_2) \triangleq \{\pmb{x} : w_1(\pmb{x}) - w_2(\pmb{x}) = 0\}$ ,  $\mathrm{dist}(x,K) \triangleq \inf_{y \in K} \|x - y\|$  for a set  $K \subseteq \mathbb{R}^d$ , and a.s. stands for the abbreviation of almost surely.

Problem Statement We assume that the zero solution of the following SDDE:

$$
\mathrm {d} \boldsymbol {x} (t) = f (\boldsymbol {x}, \boldsymbol {x} (t - \tau), t) \mathrm {d} t + g (\boldsymbol {x}, \boldsymbol {x} (t - \tau), t) \mathrm {d} B _ {t} \tag {3}
$$

is unstable, i.e.  $\lim_{t\to \infty}\pmb {x}(t;\xi)\neq \mathbf{0}$  on some set of positive measures. We aim to stabilize the zero solution with control based on neural networks (NNs). In other words, our goal is to leverage the NNs to design an appropriate controller  $\pmb {u} = (\pmb {u}_f,\pmb {u}_g)$  with  $\pmb {u}_f(\pmb {0},\pmb {0},t) = \pmb {u}_g(\pmb {0},\pmb {0},t) = \pmb{0}$  such that the controlled system

$$
\mathrm {d} \boldsymbol {x} = [ f + \boldsymbol {u} _ {f} (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau), t) ] \mathrm {d} t + [ g + \boldsymbol {u} _ {g} (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau), t) ] \mathrm {d} B _ {t} \tag {4}
$$

is stabilized to the zero solution. We call  $\pmb{u}_f: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}_+ \to \mathbb{R}^d$  as deterministic control and  $\pmb{u}_g: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}_+ \to \mathbb{R}^{d \times r}$  as stochastic control since they are integrated with  $\mathrm{dt}$  and  $\mathrm{dB}_t$ , respectively. The major difficulty of this problem comes from the non-Markovian property of SDDEs, hence we cannot apply MDP (Markov decision process)-based methods, such as reinforcement learning, to control SDDEs. Existing works prefer to learning deterministic control and regard the noise as a negative part that may destroy the natural dynamics of  $f$ . In what follows, we not only show that the deterministic control can achieve stabilization, but also that it can be done by elaborately-designed stochastic control, yielding two frameworks, the neural deterministic control (Section 3) and the neural stochastic control (Section 4). We make all our code and data available at https://anonymous.4open.science/r/SYNC-35E8.

# 3 NEURAL DETERMINISTIC CONTROL

In this section, we propose a framework of neural deterministic control (NDC) based on Theorem 2.2 to stabilize system (3) and provide experimental comparisons with traditional control methods. In addition, we analytically and experimentally investigate that the framework cannot be directly generalized to the stochastic control version. Such a problem will be addressed in the next section.

# 3.1 METHOD: LEARNING CONTROL AND AUXILIARY FUNCTIONS

The core idea of our method is that once we construct the auxiliary functions  $V$ ,  $\gamma$ ,  $w_{1}$ ,  $w_{2}$  and the neural controller  $\mathbf{u}$ , resulting in the controlled system (4) satisfying all the conditions assumed in Theorem 2.2, then the solution  $\mathbf{x}(t;\xi)$  converges to the  $\mathrm{Ker}(w_1 - w_2)$ . In particular, if we set  $\mathrm{Ker}(w_1 - w_2) = \{\mathbf{0}\}$ , the unstable zero solution of the control-free system (3) can be stabilized. To this end, we first provide the reasonable constructions of NNs to learn these candidate functions. Thus, we design the explicit form of loss function in the learning step.

Auxiliary Function We employ a multi-layer feedforward neural network, denoted by  $\mathbf{NN}(\cdot ;\theta)$ , to design all the functions. To be concrete,  $\theta_{1}$  is the parameter vector of the positive function  $V(\pmb {x},t;\theta_{1})$  and the  $L_{2}$  term  $\| \pmb {x}\| ^2$  is added to guarantee  $\lim_{\| \pmb {x}\| \to \infty}\inf_{0\leq t < \infty}V(\pmb {x},t;\theta_1) = \infty$ , that is

$$
V (\boldsymbol {x}, t; \theta_ {V}) = \mathbf {N N} (\boldsymbol {x}, t; \theta_ {V}) ^ {2} + \varepsilon \| \boldsymbol {x} \| ^ {2}, \varepsilon > 0. \tag {5}
$$

In our framework, it requires  $V \in C^{2,1}(\mathbb{R}^d \times \mathbb{R}_+)$ . We therefore choose  $C^2$  activation function of an NN, such as the hyperbolic tangent function,  $\mathrm{Tanh}(\cdot)$ . In order to design an integrable positive function  $\gamma(t)$  with an NN, we use an activation function with at most linear growth such as ReLU and multiply an exponential decay factor to the output of the NN, that is

$$
\gamma (t; \theta_ {\gamma}) = \exp (- c t) \mathbf {N} \mathbf {N} (t; \theta_ {\gamma}) ^ {2}, c > 0. \tag {6}
$$

For simplicity, we design  $w(\pmb{x}, \theta_w) = \mathbf{NN}(\pmb{x}; \theta_w)^2$  as a positive function. Additionally, we set

$$
w _ {2} = w, \quad w _ {1} = w + p (x), \quad p \geq 0, \operatorname {K e r} (p) = \{\mathbf {0} \}. \tag {7}
$$

Deterministic Control Function We first consider the deterministic control, i.e.  $\pmb{u} = (\pmb{u}_f, \pmb{0})$ . To guarantee the same zero solution of the control-free system (3) and the controlled system (4), the NDC  $\pmb{u}_f: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}^+ \to \mathbb{R}^d$  should satisfy  $\pmb{u}_f(\pmb{0}, \pmb{0}, t) = \pmb{0}$ . One feasible way to meet such a condition is to set  $\pmb{u}_f(\pmb{x}, \pmb{y}, t) = \mathbf{NN}(\pmb{x}, \pmb{y}, t; \theta_f) - \mathbf{NN}(\pmb{0}, \pmb{0}, t; \theta_f)$  or  $\pmb{u}_f(\pmb{x}, \pmb{y}, t) = \mathrm{diag}(\pmb{x})\mathbf{NN}(\pmb{x}, \pmb{y}, t; \theta_f)$ . Here,  $\mathrm{diag}(\pmb{x})$  is a diagonal matrix with its  $i$ -th diagonal element being  $x_i$ .

Loss Function Once the learned functions  $V, \gamma, w_{1}, w_{2}$  and  $\mathbf{u}$  with the coefficient functions,  $f_{\mathbf{u}} \triangleq f + \mathbf{u}$  and  $g$ , in the controlled system (4), meet all the conditions assumed in Theorem 2.2, the stability of zero solution is naturally assured. To achieve this, we demand a suitable loss function to evaluate the likelihood that those conditions are satisfied. It can be seen from our construction that the only condition needed to be satisfied is  $\mathcal{L}V(\mathbf{x}, \mathbf{y}, t) \leq \gamma(t) - w_{1}(x) + w_{2}(y)$ . Hence, we define LaSalle's loss function for the controlled system (4) as follows.

Definition 3.1 (LaSalle's Loss) Consider the above parameterized candidate functions  $V, \gamma, w_{1}, w_{2}$  and a controller  $\mathbf{u}_{f}$  for the controlled system (4). Then, LaSalle's loss is defined as

$$
L _ {\mu , \varepsilon , c, p} (\boldsymbol {\theta} _ {V}, \boldsymbol {\theta} _ {\gamma}, \boldsymbol {\theta} _ {w}, \boldsymbol {\theta} _ {f}) = \mathbb {E} _ {(\boldsymbol {x}, \boldsymbol {y}, t) \sim \mu} [ \max  (0, \mathcal {L} V (\boldsymbol {x}, \boldsymbol {y}, t) - \gamma (t) + w _ {1} (\boldsymbol {x}) - w _ {2} (\boldsymbol {y})) ],
$$

where the state variable  $x$  obeys the distribution  $\mu$ . In practice, we consider the following empirical loss function through the Monte Carlo sampling

$$
L _ {N, \varepsilon , c, p} \left(\boldsymbol {\theta} _ {V}, \boldsymbol {\theta} _ {\gamma}, \boldsymbol {\theta} _ {w}, \boldsymbol {\theta} _ {f}\right) = \frac {1}{N} \sum_ {i = 1} ^ {N} \max  \left(0, \mathcal {L} V \left(\boldsymbol {x} _ {i}, \boldsymbol {y} _ {i}, t _ {i}\right) - \gamma \left(t _ {i}\right) + w _ {1} \left(\boldsymbol {x} _ {i}\right) - w _ {2} \left(\boldsymbol {y} _ {i}\right)\right), \tag {8}
$$

where  $\{\pmb{x}_i,\pmb{y}_i,t_i\}_{i = 1}^N$  are sampled from the distribution  $\mu = \mu (\Omega)$  and  $\Omega$  is some closed domain in  $\mathbb{R}^d\times \mathbb{R}^d\times \mathbb{R}_+$ .

In summary, the developed NDC framework is shown in Algorithm 1 in Appendix A.3.1.

Remark 3.1 The proposed NDC framework can be easily applied to the autonomous SDDE  $\mathrm{d}\pmb{x}(t) = f(\pmb{x},\pmb{x}(t - \tau))\mathrm{d}t + g(\pmb{x},\pmb{x}(t - \tau))\mathrm{d}B_t$ . In particular, one can simply consider the autonomous auxiliary function  $V$  and the control function, and set  $\gamma (t) = 0$ . For sample distribution  $\mu (\Omega)$ , here we select the uniform distribution on a large enough closed region  $\Omega$  as used in (Han et al., 2016; Chang et al., 2019), and we provide further analysis about the influence of  $\mu$  in Appendix A.2.1.

# 3.2 NUMERICAL AND ANALYTICAL INVESTIGATIONS

Comparison with Existing Methods Recent works on controlling time-delayed systems mainly focus on elaborately designing the analytical form of control to satisfy the conditions in the LaSalle-Type Theorem 2.2 (Lin & He, 2005; Xu et al., 2014), or simultaneously designing control and the Lyapunov function to satisfy the conditions based on the Lyapunov theory (Yu & Cao, 2007). It should be noted that all these methods require a delicate design of functions for specific dynamics, and thus are limited in practical application for controlling general time-delayed systems. However, our neural method leverages neural networks to automatically learn the control policies, and can be easily applied in any kinds of time-delayed systems having stochastic settings. In Figure 3, we numerically compare the NDC and a baseline, the linear control (LC) proposed in (Lin & He, 2005), on a noised driving-response Chua's circuit. Here, Chua's circuit is a third-order autonomous dynamical system with only one nonlinear element, producing typical chaotic dynamics (Matsumoto, 1984). In the experiment, we show that the NDC can find the neural control for the response system  $\pmb{y} = (y_{1}, y_{2}, y_{3})$  with the autonomous and even the nonautonomous time-delay noise while the nonautonomous time-delay noise is beyond the concern in (Lin & He, 2005). The configuration of the experiment is described in Appendix A.3.4.

Failure in Finding Stochastic Control As we can see that the NDC performs well, a natural idea is to utilize the noise part to achieve the stabilization of the SDDE (3). To explore this idea, we adopt the same NN of  $\pmb{u}_f$ , design  $\pmb{u}_g = \mathbf{NN}(\pmb{x},\pmb{y},t;\pmb{\theta}_g)$ , and train its parameters  $\pmb{\theta}_g$  with LaSalle's loss (8). However, in Figure 2, we show that the loss cannot converge to zero

![](images/f84cc12c85a7e298fa3ba3693e7ac4cd30652e9df9ba73aae3bc867edd102fa2.jpg)  
Figure 2: Training loss for the 1-D SDDE.

in controlling a simple 1-D toy system via the stochastic controller  $\pmb{u}_g$ :  $\mathrm{d}x(t) = [x(t) + x(t - \tau)]\mathrm{d}t+$

![](images/49bc56e690430b2fbc1367265124f495ccdc822719450f594109edfdeb72296a.jpg)  
(a)

![](images/bf02c6d8bcbff977e7c5c7c782036f8404703faf3286fc473046acd05d3efa8c.jpg)  
Figure 3: (a) The original driving-response model, (b) the controlled orbits under LC and NDC, (c) the time trajectory of  $y_{2}$  with autonomous noise, and (d) nonautonomous noise. The solid lines are obtained through averaging the 10 sampled trajectories, while the shaded areas stand for the standard errors.  
(b)

![](images/fbb15e859b657433077d63d136e5b9f36a342ca028f16f8c75e79fede27a9b96.jpg)  
(c)

![](images/5cca6a70aec0755853d59251503aa525038fe6c6b6a736ef20005d543e8a1fc6.jpg)  
(d)

$[x(t - \tau) + u_g(x(t), x(t - \tau); \theta_g)] \mathrm{d}B_t$ . Actually, this phenomena can be analytically explained. Notice that  $\pmb{\theta}_g$  arises in loss function as a quadratic term  $l(\pmb{\theta}_g) = \frac{1}{2} \mathrm{Tr}[u_g^\top \mathcal{H}V \pmb{u}_g]$  according to Eq. (2), the sign of this term depends on the convexity of  $V$ , i.e. the maximum eigenvalue's sign of  $\mathcal{H}V$ . Nevertheless, the positive function  $V$  with  $\lim_{\| \pmb{x} \| \to \infty} V(\pmb{x}, t) = \infty$  implies  $l(\pmb{\theta}_g) \geq 0$  at most time. Hence, the ideal case  $l(\pmb{\theta}_g) = 0$  in the training procedure is equivalent to  $\pmb{u}_g = 0$ , which means that we are unable to learn a stochastic controller under LaSalle's loss (8) satisfying the sufficient conditions assumed in Theorem 2.2.

Drawbacks of  $L^2$  regularization in  $V$  Adding  $L^2$  regularization to objective functions is a classical operation to avoid over-fitting (Ying, 2019) and guarantee the positive definiteness (Gallieri et al., 2019). However, the explicit form  $\varepsilon \| x\|^2$  may fail in learning an effective neural control as this function cannot be the candidate  $V$  function in some cases (Zhang et al., 2022). The following example illustrates this point.

Example 3.2 Consider a 2-D SDDE as follows:

$$
\mathrm {d} x _ {1} (t) = x _ {2} (t) \mathrm {d} t + \frac {1}{2} x _ {1} (t - 1) \mathrm {d} B _ {1} (t), \mathrm {d} x _ {2} (t) = [ - 2 x _ {1} (t) - x _ {2} (t) ] \mathrm {d} t + x _ {1} (t) \mathrm {d} B _ {2} (t)
$$

In Appendix A.1.3, the solution of this system is validated to satisfy  $\lim_{t\to \infty}\pmb {x}(t;\xi) = \mathbf{0}$  a.s. with any initial data  $\xi \in C_{\mathcal{F}_0}([-1,0];\mathbb{R}^2)$ ; however,  $k\| \pmb {x}\| ^2$  for any  $k\in \mathbb{R}_+$  cannot be a useful auxiliary  $V$  function to identify the sufficient conditions in Theorem 2.2.

# 4 NEURAL STOCHASTIC CONTROL

To address the illustrated problems above, we design a new concise framework for fast learning a stochastic controller, called the neural stochastic control (NSC). To this end, we first provide the following theoretical result on stabilization of general stochastic functional differential equations (SFDEs) with the proof provided in Appendix A.1.4.

Theorem 4.1 Consider the SFDE  $\mathrm{d}\pmb{x}(t) = F(\pmb{x}_t,t)\mathrm{d}t + G(\pmb{x}_t,t)\mathrm{d}B(t)$ , with  $F,G$  being locally Lipschitzian functions,  $F(\mathbf{0},t) = \mathbf{0}$ , and  $G(\mathbf{0},t) = \mathbf{0}$ . For every  $M > 0$ , assume that  $\min_{\| \pmb{x}_t(0)\| = M}\| \pmb{x}_t(0)^\top G(\pmb{x}_t,t)\| >0$ . If there exists a number  $\alpha \in (0,1)$  such that

$$
\left\| \boldsymbol {x} _ {t} (0) \right\| ^ {2} \left(2 \left\langle \boldsymbol {x} _ {t} (0), F (\boldsymbol {x} _ {t}, t) \right\rangle + \left\| G (\boldsymbol {x} _ {t}, t) \right\| _ {\mathrm {F}} ^ {2}\right) - (2 - \alpha) \left\| \boldsymbol {x} _ {t} (0) ^ {\top} G (\boldsymbol {x} _ {t}, t) \right\| ^ {2} \leq 0, \tag {9}
$$

for  $\pmb{x}_t \in C([-\tau, 0], \mathbb{R}^d)$ , where  $\pmb{x}_t(s) = \pmb{x}(t + s)$  for  $s \in [-\tau, 0]$ . Then, the solution of the SFDE satisfies  $\lim_{t \to \infty} \pmb{x}(t; \xi) = 0$  a.s. for any  $\xi \in C_{\mathcal{F}_0}([-\tau, 0]; \mathbb{R}^d)$ .

Remark 4.2 The SFDE in Theorem 4.1 is formulated in a very general type, including the SDDE  $\mathrm{d}\pmb {x}(t) = F(\pmb {x}(t),\pmb {x}(t - \tau_1),\dots ,\pmb {x}(t - \tau_q),t)\mathrm{d}t + G(\pmb {x}(t),\pmb {x}(t - \tau_1),\dots ,\pmb {x}(t - \tau_q),t)\mathrm{d}B_t$  with  $\tau_{1} < \tau_{2} < \dots < \tau_{q}\in [0,\tau ]$ . This indicates that our framework can be generalized to stabilize the SDDEs with multiple delays and even more general SFDEs as well.

In light of Theorem 4.1, we establish a more general framework for learning a neural controller of system (4) with the form  $\pmb{u} = (\pmb{u}_f, \pmb{u}_g)$  designed in the same NN architecture as the one used in the NDC framework. We focus on stochastic control with  $\pmb{u}_f = \mathbf{0}$  and provide more control combinations in Appendix A.3.3, whereas the loss function is differently designed as follows.

Definition 4.1 (Asymptotic Loss) Utilizing the notations in Definition 3.1 and  $g_{\pmb{u}} = g + \pmb{u}_{g}$ , the loss function for the controlled system (4) with the controller  $\pmb{u}$  is defined as:

$$
L _ {\mu , \alpha} (\boldsymbol {\theta}) = \mathbb {E} _ {(\boldsymbol {x}, \boldsymbol {y}, t) \sim \mu} \left[ \max  \left(0, (\alpha - 2) \| \boldsymbol {x} ^ {\top} g _ {\boldsymbol {u}} (\boldsymbol {x}, \boldsymbol {y}, t) \| ^ {2} + \| \boldsymbol {x} \| ^ {2} (2 \langle \boldsymbol {x}, f (\boldsymbol {x}, \boldsymbol {y}, t) \rangle + \| g _ {\boldsymbol {u}} (\boldsymbol {x}, \boldsymbol {y}, t) \| _ {\mathrm {F}} ^ {2})\right) \right], \tag {10}
$$

where  $\pmb{\theta} = (\pmb{\theta}_f, \pmb{\theta}_g)$ . Akin to Definition 3.1, we use the empirical loss function for training.

Here,  $\alpha$  is an adjustable parameter, which is related to the convergence rate and the control energy. We further discuss the design of the asymptotic loss in Appendix A.2.2 and numerically investigate the role of  $\alpha$  in Appendix A.4.1. We summarize the framework in Algorithm 3 in Appendix A.3.1. And we further compare the computational complexity in Appendix A.3.2.

# 4.1 EXPERIMENTS OF THE COMBINATION METHODS

We compare our neural control methods on a noise-perturbed kinematic bicycle model for car-like vehicles (Rajamani, 2011) in terms of the convergence time and the energy cost, which are two important indexes to measure the quality of a controller (Yan et al., 2012; Li et al., 2017; Sun et al., 2017). To quantify the energy cost in the control process, we

Table 1: Results on kinematic bicycle model.  

<table><tr><td></td><td>Tt</td><td>E0.001</td><td>Nd</td><td>E[τ0.001]</td></tr><tr><td>NDC</td><td>1028.81s</td><td>102.17</td><td>6.3e-4</td><td>1.81</td></tr><tr><td>NSC</td><td>59.80s</td><td>62.10</td><td>4.0e-7</td><td>0.29</td></tr><tr><td>QP</td><td>-</td><td>-</td><td>0.016</td><td>&gt;5</td></tr></table>

first denote the stopping time by  $\tau_{\epsilon} \triangleq \inf \{t > 0 : \| \pmb{x}(t) \| = \epsilon\}$  and then denote the energy cost by  $\mathcal{E}_{\epsilon} \triangleq \mathbb{E}\left[\int_0^{\tau_{\epsilon}}\left(\|\pmb{u}_f\|^2 + \|\pmb{u}_g\|^2\right)\mathrm{d}t\right]$ . We approximate this expectation value by the empirical value as  $\frac{1}{N}\sum_{i=1}^{N}\int_{0}^{\tau_{\epsilon}^i}\left(\|\pmb{u}_f^i\|^2 + \|\pmb{u}_g^i\|^2\right)\mathrm{d}t$  through the Monte Carlo sampling. We show the results in Figure 4 and Table 1. Table 1 includes the training time (Tt), empirical energy cost  $\mathcal{E}_{0.001}$ , nearest distance (Nd) between the bicycle and target position, empirical expectation  $\mathbb{E}[\tau_{0.001}]$  of different methods. We provide more experimental details in Appendix A.3.5. We can see that the ranking of the comprehensive performance is that NSC is greater than NDC than QP. This means that we can really benefit from introducing noise as control. An explanation for this phenomena is that when we regard energy cost as objective function to be minimized, randomness can lead this functional to the shortest path quicker than the deterministic control, just like stochastic gradient descent outperforms the full-batch gradient descent. We show the NSC can enlarge the region of attraction of 100-D gene regulatory networks in Appendix A.4.2.

Uncontrollable Fluctuation Although the neural stochastic method we propose outperforms the control methods including deterministic control, there exists an obvious disadvantage that the method can cause uncontrollable fluctuation due to the stochasticity. However, we always want to bound this perturbation in practical application owing to physical and engineering restrictions in real world. We tackle this safety guarantee problem for our methods in Section 6.

![](images/2a244b00c2b26dc368280b7e2e3c9c6c589e5f12bc96a1cffa769bf3d85a7be8.jpg)  
Figure 4: (Left) A schematic diagram of the kinematic bicycle model. (Right) Time trajectories of the state variables  $x, y$  of the kinematic bicycle under different control cases. The solid lines are obtained through averaging the 10 sampled trajectories, while the shaded areas stand for the standard errors.

![](images/ea40583a884f33f036221fba74be9e37f1733b2044847b0de6dd2fe8a271c5b8.jpg)

![](images/fae5d9114cff38f45ae014b52bd148badded2bb00cadd8cc9fd3facb7e2c96ea.jpg)

![](images/edb2575c497010c35266db8db78e77a96e87817fe641918e27c749a75932c156.jpg)

![](images/75d85bbd2dbaaf51d8514db8a1b72865c5371483ff821820c92afe9bf376c95a.jpg)

# 5 THEORETICAL RESULTS FOR NDC AND NSC

We have mentioned the stopping time and the energy cost in section 4.1 and numerically compare the proposed neural controllers with these indexes. These two indexes are the classic factors to measure the performance of the controller (Sun et al., 2017). In this section, we provide theoretical estimation for the upper bound of these indexes in controlled dynamics with NDC/NSC. To begin with, we choose suitable activation function in our NN control such that the neural control function  $u(x)$  is Lipschitz continuous with Lipschitz constant  $k_{u}$  (Fazlyab et al., 2019; Aziznejad et al., 2020). Then we have the following two theorems and we provide the proofs in Appendix A.1.7,A.1.8.

Theorem 5.1 (Estimation for NDC) Consider the SDDE with NDC controller as

$$
\mathrm {d} \boldsymbol {x} (t) = \left(f (\boldsymbol {x}, \boldsymbol {x} (t - \tau)) + \boldsymbol {u} _ {f} (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau)) \mathrm {d} t + g (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau)) \mathrm {d} B _ {t}, \right. \boldsymbol {x} (0) = \boldsymbol {x} _ {0} \in \mathbb {R} ^ {d},
$$

where  $\| f(\pmb{x}, \pmb{y}) - f(\bar{\pmb{x}}, \bar{\pmb{y}}) \| \vee \| \pmb{u}_f(\pmb{x}, \pmb{y}) - \pmb{u}_f(\bar{\pmb{x}}, \bar{\pmb{y}}) \| \leq L (\| \pmb{x} - \bar{\pmb{x}} \| + \| \pmb{y} - \bar{\pmb{y}} \|)$ . Assume the controlled system satisfies the similar conditions in Theorem 2.2 and Remark 3.1 with  $\mathrm{Ker}(w_1 - w_2) = 0$ . Denote stopping time as  $\eta_{\varepsilon} = \inf \{t > 0 : \| \pmb{x}(t) \| = \varepsilon \}$ , the corresponding energy cost in the control process with  $\epsilon < \| \pmb{x}_0 \|$  as  $\mathcal{E}(\eta_{\varepsilon}, T) = \mathbb{E}\left[\int_0^{\eta_{\varepsilon} \wedge T} \| \pmb{u}(\pmb{x}(s), \pmb{x}(s - \tau)) \|^2 \, \mathrm{d}s\right]$  and under the same notations in Theorem 2.2, we have

$$
\left\{ \begin{array}{l} \mathbb {E} [ \eta_ {\epsilon} ] \leq T _ {\epsilon} = \frac {V (\boldsymbol {x} _ {0}) - \min  _ {\| \boldsymbol {x} \| = \varepsilon} V (\boldsymbol {x}) + \int_ {- \tau} ^ {0} w _ {2} (\xi (s)) \mathrm {d} s}{\min  _ {\| \boldsymbol {x} \| \geq \varepsilon} (w _ {1} (\boldsymbol {x}) - w _ {2} (\boldsymbol {x}))}, \\ \mathcal {E} (\eta_ {\epsilon}, T _ {\epsilon}) \leq \frac {k _ {\boldsymbol {u}} ^ {2} C _ {0}}{2 (L ^ {2} + L + k _ {\boldsymbol {u}})} \left[ \exp \left(4 (L ^ {2} + L + k _ {\boldsymbol {u}}) T _ {\varepsilon}\right) - 1 \right] + \int_ {- \tau} ^ {0} k _ {\boldsymbol {u}} ^ {2} \xi^ {2} (s) \mathrm {d} s. \end{array} \right.
$$

where  $C_0 = \| \pmb{x}_0\|^2 + (2L^2 + L + k_{\pmb{u}})\int_{-\tau}^{0}\xi(s)^2\mathrm{d}s$  and  $\xi \in C[-\tau, 0]$  is the initial data.

Theorem 5.2 (Estimation for NSC) Consider the SDDE with NSC controller as

$$
\mathrm {d} \boldsymbol {x} (t) = f (\boldsymbol {x}, \boldsymbol {x} (t - \tau)) \mathrm {d} t + (g (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau)) + \boldsymbol {u} _ {g} (\boldsymbol {x} (t), \boldsymbol {x} (t - \tau)) \mathrm {d} B _ {t}, \quad \boldsymbol {x} (0) = \boldsymbol {x} _ {0} \in \mathbb {R} ^ {d},
$$

where  $f, g$  are the same as those in Theorem 6.2. Assume the controlled system satisfies the similar conditions in Theorem 4.1, under the same notations in Theorem 4.1, if the term in (9) further satisfies  $\max_{\| \boldsymbol{x}_t(0)\| \geq \varepsilon}\| \boldsymbol{x}_t(0)\|^{\alpha - 4}(\| \boldsymbol{x}_t(0)\|^2(2\langle \boldsymbol{x}_t(0), f(\boldsymbol{x}_t) \rangle + \| G(\boldsymbol{x}_t)\|_{\mathrm{F}}^2) - (2 - \alpha)\| \boldsymbol{x}_t(0)^\top G(\boldsymbol{x}_t)\|^2) = -\delta_\varepsilon < 0$  with  $G = g + \boldsymbol{u}_g$ , we have

$$
\left\{ \begin{array}{l} \mathbb {E} [ \eta_ {\epsilon} ] \leq T _ {\epsilon} = \frac {2 \left(\| \boldsymbol {x} _ {0} \| ^ {\alpha} - \varepsilon^ {\alpha}\right)}{\alpha \cdot \delta_ {\varepsilon}}, \\ \mathcal {E} (\eta_ {\epsilon}, T _ {\epsilon}) \leq \frac {k _ {\boldsymbol {u}} ^ {2} C _ {1}}{2 (2 L ^ {2} + L + k _ {\boldsymbol {u}} ^ {2})} \left[ \exp \left(4 (2 L ^ {2} + L + k _ {\boldsymbol {u}} ^ {2}) T _ {\varepsilon}\right) - 1 \right] + \int_ {- \tau} ^ {0} k _ {\boldsymbol {u}} ^ {2} \xi^ {2} (s) \mathrm {d} s. \end{array} \right.
$$

where  $C_1 = \| \pmb{x}_0\|^2 + (4L^2 + L + 2k_u^2)\int_{-\tau}^{0}\xi(s)^2\mathrm{d}s$  and  $\xi \in C[-\tau, 0]$  is the initial data.

We can design the neural network structure according to these theoretical results. Here we only analyse  $T_{\varepsilon}$  because the energy cost  $\mathcal{E}(\eta_{\varepsilon},T_{\varepsilon})$  explicitly depends on  $T_{\varepsilon}$ . First, the upper bound for  $\mathbb{E}[\eta_{\varepsilon}]$  of NDC implies that the convergence time decrease as the slope of  $w = w_{1} - w_{2}$  near the origin grows due to the fact that  $w^{\prime}(\mathbf{0})\approx w(\varepsilon) / \varepsilon$ , and the same effect holds for  $V$ . Hence, we can construct  $w_{1,2}$  and  $V$  as neural networks with steeper slope at the origin to accelerate the control process, and thus reduce the upper bound of energy cost, for NDC. Secondly, the time upper bound of NSC is directly relates to the hyperparameter  $\alpha$  in the training period, so we can choose  $\alpha^{*} = \arg \min_{\alpha}(\| x_0^\alpha -\varepsilon^\alpha \|) / \alpha$  to obtain the optimal NSC controller with the least upper bound of convergence time and energy cost. We numerically investigate the influence of  $\alpha$  in Appendix A.4.1.

# 6 SAFETY GUARANTEE FOR SDDES

In this section, we provide safety guarantee for our NDC and NSC, that is, the SYNC. We first extend the recent results on stochastic control barrier functions (Clark, 2019) to the SDDE, deriving constraints on the stochastic control. This guarantees that the process  $\pmb{x}(t;\xi)$  satisfies the safety constraint, i.e.,  $\pmb{x}(t;\xi) \in \mathrm{int}(\mathcal{C})$  for all  $t$  with the initial value  $\xi(0) \in \mathrm{int}(\mathcal{C})$ . The set  $\mathcal{C}$  is defined by a locally Lipschitzian function  $h: \mathbb{R}^d \to \mathbb{R}$  as  $\mathcal{C} = \{\pmb{x}: h(\pmb{x}) \geq 0\}$ . We summarize the extension results in the following proposition, and include the proof of this proposition in Appendix A.1.5.

Proposition 6.1 Let the function  $\mathcal{B}:\mathbb{R}^d\to \mathbb{R}$  be locally Lipschitz and twice-differentiable on int(C). If there exists class-  $K$  functions  $\alpha_{1}(\pmb {x})$ $\alpha_{2}(\pmb {x})$  and  $\alpha_{3}(\pmb {x})$  such that,

$$
\frac {1}{\alpha_ {1} (h (\boldsymbol {x}))} \leq \mathcal {B} (\boldsymbol {x}) \leq \frac {1}{\alpha_ {2} (h (\boldsymbol {x}))}, a n d \mathcal {L B} (\boldsymbol {x}, \boldsymbol {y}, t) \leq \alpha_ {3} (h (\boldsymbol {x})),
$$

where class-  $K$  denotes the set of all functions:  $\mathbb{R}_+ \to \mathbb{R}_+$ , which are continuous, strictly increasing and vanishing at zero. Then  $\mathcal{B}(x)$  is called a stochastic control barrier function (SCBF). Suppose there exists a SCBF for the SDDE (1). Then for all  $t$ ,  $\mathbb{P}\big(\pmb{x}(t) \in \mathrm{int}(\mathcal{C})\big) = 1$ , provided that  $\pmb{x}(0) \in \mathrm{int}(\mathcal{C})$ .

Baseline With Proposition 6.1 and Theorem 2.2, the traditional deterministic control methods based on the Quadratic Program (QP) in (Fan et al., 2020; Sarkar et al., 2020) can be easily applied to the SDDE. We use this QP method as baseline and the specific algorithm is shown in Appendix A.3.1. We further use the classic MPC method as a baseline control.

Proposed SYNC A natural idea is to integrate Proposition 6.1 into our proposed neural framework, but there are three inequality constraints in this proposition, making the NN constructions complicated and hard to train. To simplify the construction, we propose the following new Theorem for safety guarantee, which is a significantly novel contribution to the existing barrier function theory.

Theorem 6.2 For stochastic functional differential equation  $\mathrm{d}\pmb{x}(t) = F(\pmb{x}_t,t)\mathrm{d}t + G(\pmb{x}_t,t)\mathrm{d}B(t)$ , with  $F, G$  satisfying locally Lipschitz condition and locally linear growth condition,  $\pmb{x}_t(s) = \pmb{x}(t + s)$  for  $s\in [-\tau ,0]$ , if there exist a number  $\alpha \in (0,1)$  and a class- $K$  function  $\lambda (x)$  such that  $2\mathcal{B}\mathcal{L}\mathcal{B} - (1 - \alpha)\| \nabla \mathcal{B}^{\top}g\|^{2}\leq 0$  for  $\pmb {x}\in \mathbb{R}^{d}$ , where  $\mathcal{B}(\pmb {x}) = \frac{1}{\lambda(h(\pmb{x}))}$ . Then, the solution of this equation satisfies  $\mathbb{P}(\pmb {x}(t;\xi)\in \operatorname {int}(\mathcal{C})) = 1$  for any  $\xi \in C_{\mathcal{F}_0}([- \tau ,0];\mathbb{R}^d)$  with  $\xi (0)\in \operatorname {int}(\mathcal{C})$ .

We provide the proof of this theorem in Appendix A.1.6. Now we only need to construct a neural candidate class-K function  $\lambda$  and add the condition in Theorem 6.2 to loss function in our aforementioned framework. Here we utilize the input convex neural network (Amos et al., 2017a) to insure  $\lambda(x; \theta_{\lambda}) = \mathrm{NN}(x; \theta_{\lambda}) - \mathrm{NN}(0; \theta_{\lambda})$  is a class-K function, and we define the loss function as follows.

Definition 6.1 (Barrier Loss) Provided with the fixed function  $h(\boldsymbol{x})$  and the barrier function  $\mathcal{B}(\boldsymbol{x}) = \frac{1}{\lambda(h(\boldsymbol{x}))}$ , we define the loss function for the controlled system (4) as:

$$
L _ {\mu , \alpha} (\boldsymbol {u}) = \mathbb {E} _ {(\boldsymbol {x}, \boldsymbol {y}) \sim \mu} \left[ \max  \left(0, 2 \mathcal {B} (\boldsymbol {x}) \mathcal {L B} (\boldsymbol {x}, \boldsymbol {y}, t) - (1 - \alpha) \| \nabla \mathcal {B} ^ {\top} (\boldsymbol {x}) g _ {u} (\boldsymbol {x}, \boldsymbol {y}, t) \| ^ {2}\right) \right] \tag {11}
$$

and, still, we use the empirical loss through the Monte Carlo sampling in our training. A significant difference is that here we pick up the uniform distribution as  $\mu = \mu (\mathrm{int}(\mathcal{C}))$

# 6.1 EXPERIMENTAL COMPARISON

We investigate the ability of the proposed safe control method to limit the fluctuation in the control process on the task of control noise-perturbed inverted pendulum, which is a standard nonlinear control problem for testing different control methods (Anderson, 1989; Huang & Huang, 2000). Mathematically, the pendulum has two state variables: the angular position  $\theta$ , that is the angle deviating from the vertical position  $\pmb{x} = \mathbf{0}$ , and angular velocity  $\dot{\theta}$ . The external effects on the pendulum can be characterized by a time-delay noise. Now, we apply the safe method to steer the system to the inverted position  $\pmb{x} = \mathbf{0}$  without rotating a full circle, i.e.  $|\theta| \leq 2\pi$ . The results are shown in Figure 5 and the experimental details are provided in Appendix A.3.6. We observed that the safe control method significantly outperms the baseline and the stochastic control method in terms of stabilization and safety guarantee.

Finally, notice that the Barrier Loss defined in Definition 6.1 computes the Hessian matrix for the barrier function  $B(\pmb{x})$ . This indicates that we should trade off between the computational cost and the safety for specific tasks.

# 7 RELATED WORKS

- Stability Theory of SDDEs The early endeavors to develop the stability theory for SDDEs were attributed to (Mao, 1999; 2002) inspired by LaSalle's theory (LaSalle, 1968). The subsequent

![](images/3fbd00007a0e2105b8ef677734af32c2d3042315122c6a2169cfc951c6f02c27.jpg)  
Figure 5: Schematic diagram of inverted pendulum task (a). The  $\theta$  component of the original system (b), under baseline control (c), under NSC (d), under our proposed safe control (e). The solid lines are obtained through averaging the 5 sampled trajectories, while the shaded areas stand for the standard errors.

![](images/8f3443a2e48a42bc7c116682a9dba718a6fa7e700bc4c550880eb4ecf1976ed3.jpg)

![](images/f31b69394dc42024fd87596b4104f3beb33e7f46f2510d705216a646fe5aa560.jpg)

![](images/9c1f5604e11cd141f13be8bf84be6e5cc1fdcdf26693cda6c5dfa18922a548b9.jpg)

![](images/c7965b10eb5a984f1b1aac7fae87d4e8a43feb4d0549578853283a85a868e57a.jpg)

developments have been systematically and fruitfully achieved in the last twenty years in the control community Appleby (2003); Song et al. (2014); Liu et al. (2016); Zhu (2018); Peng et al. (2021). These works reveal the positive effect of multiplicative noise to the stochastic dynamics with delays. These, therefore, motivate us to develop only neural stochastic control to stabilize dynamical systems in this work.

- Finding Stabilization Controller Traditional control methods focus on transforming control criteria, such as the control Lyapunov functions (CLFs), into the QP (Fan et al., 2020; Sarkar et al., 2020) or the semi-definite planning (SDP) problems (Henrion & Garulli, 2005; Jarvis-Wloszek et al., 2003; Parrilo, 2000) to find optimal control iteratively. These methods have high computational complexity since they cannot give the closed form of the control. Hence, machine-learning based control methods have been introduced to improve the generalization and efficiency of the original convex optimal problems (Khansari-Zadeh & Billard, 2014; Ravanbakhsh & Sankaranarayanan, 2019; Gurriet et al., 2018). From the Model Predictive Control based learning (Wagener et al., 2019; Williams et al., 2018), to the neural Lyapunov control (Chang et al., 2019), and even to the recently neural stochastic control (Zhang et al., 2022), all these approaches consider dynamics without time-delay. In comparison, our work focuses much on learning control policies for more general dynamics with both time-delay and stochastic settings.

- Theory and Application of Control Barrier Function The barrier function method has been extensively researched in the problem of safety verification of controlled dynamics in both deterministic setting (Prajna & Jabbabaie, 2004; Jankovic, 2018) and stochastic setting (Prajna et al., 2004; Clark, 2019; 2021). However, the existing works always require barrier function to satisfy many inequality constraints and do not consider nonlinear systems with time-delay in either drift term or diffusion term. In comparison, we propose a new provable simplified theorem to achieve safety guarantee for SFDE(SDDE) with only one inequality constraint. Existing works for constructing barrier functions in application typically based on quadratic programming (Ames et al., 2014; 2016; Khojasteh et al., 2020; Fan et al., 2020). Machine learning methods have also be introduced in safe control fields in (Robey et al., 2020; Dean et al., 2020; Taylor et al., 2020), but these works mainly focus on learning model uncertainty or barrier function instead of control policy. To the best of our knowledge, our neural safe methods is the first result that simultaneously learn safe control policy and barrier functions from data.

# 8 DISCUSSION

We have proposed SYNC, a novel control family including two frameworks to learn control policies for nonlinear systems with time-delay in stochastic setting with safe guarantee. To this end, We prove two new theorems on stability and safety criteria for SFDEs that are more concise than the existing theories. The frameworks significantly simplify the process of control design, and we theoretically and numerically investigate the propose methods in terms of energy cost and convergence time with comparison of the existing methods. Furthermore, our frameworks can easily generalize to find control policies for SFDEs with the theorems we proved. Finally, we provide some limitations of the SYNC in Appendix A.2, which need in-depth investigation in future works.

# REFERENCES

Uri Alon. An introduction to systems biology: design principles of biological circuits. Chapman and Hall/CRC, 2006.  
Aaron D Ames, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs with application to adaptive cruise control. In 53rd IEEE Conference on Decision and Control, pp. 6271-6278. IEEE, 2014.  
Aaron D Ames, Xiangru Xu, Jessy W Grizzle, and Paulo Tabuada. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 62(8): 3861-3876, 2016.  
Brandon Amos, Lei Xu, and J Zico Kolter. Input convex neural networks. In International Conference on Machine Learning, pp. 146-155. PMLR, 2017a.  
Brandon Amos, Lei Xu, and J Zico Kolter. Input convex neural networks. In International Conference on Machine Learning, pp. 146-155. PMLR, 2017b.  
Charles W Anderson. Learning to control an inverted pendulum using neural networks. IEEE Control Systems Magazine, 9(3):31-37, 1989.  
John AD Appleby. Stabilisation of functional differential equations by noise. Systems & Control Letters, 2003.  
Shayan Aziznejad, Harshit Gupta, Joaquim Campos, and Michael Unser. Deep neural networks with trainable activations and controlled lipschitz constant. IEEE Transactions on Signal Processing, 68:4688-4699, 2020.  
Eduardo F Camacho and Carlos Bordons Alba. Model predictive control. Springer science & business media, 2013.  
Ya-Chien Chang, Nima Roohi, and Sicun Gao. Neural lyapunov control. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3245-3254, 2019.  
Andrew Clark. Control barrier functions for complete and incomplete information stochastic systems. In 2019 American Control Conference (ACC), pp. 2928-2935. IEEE, 2019.  
Andrew Clark. Control barrier functions for stochastic systems. Automatica, 130:109688, 2021.  
Sarah Dean, Andrew J Taylor, Ryan K Cosner, Benjamin Recht, and Aaron D Ames. Guaranteeing safety of learned perception modules via measurement-robust control barrier functions. arXiv preprint arXiv:2010.16001, 2020.  
David D Fan, Jennifer Nguyen, Rohan Thakker, Nikhilesh Alatur, Ali-akbar Agha-mohammadi, and Evangelos A Theodorou. Bayesian learning-based adaptive control for safety critical systems. In 2020 IEEE international conference on robotics and automation (ICRA), pp. 4093–4099. IEEE, 2020.  
Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George Pappas. Efficient and accurate estimation of lipschitz constants for deep neural networks. Advances in Neural Information Processing Systems, 32, 2019.  
Marco Gallieri, Seyed Sina Mirrazavi Salehian, Nihat Engin Toklu, Alessio Quaglino, Jonathan Masci, Jan Koutnik, and Faustino Gomez. Safe interactive model-based learning. arXiv preprint arXiv:1911.06556, 2019.  
Qian Guo, Xuerong Mao, and Rongxian Yue. Almost sure exponential stability of stochastic differential delay equations. SIAM Journal on Control and Optimization, 54(4):1919-1933, 2016.  
Thomas Gurrier, Andrew Singletary, Jacob Reher, Laurent Ciarletta, Eric Feron, and Aaron Ames. Towards a framework for realizable safety critical control through active set invariance. In 2018 ACM/IEEE 9th International Conference on Cyber-Physical Systems (ICCPs), pp. 98-106. IEEE, 2018.

Jiequn Han et al. Deep learning approximation for stochastic control problems. arXiv preprint arXiv:1611.07422, 2016.  
Didier Henrion and Andrea Garulli. Positive Polynomials in Control, volume 312. Springer Science & Business Media, 2005.  
Shiuh-Jer Huang and Chien-Lo Huang. Control of an inverted pendulum using grey prediction model. IEEE Transactions on Industry Applications, 36(2):452-458, 2000.  
Mrdjan Jankovic. Control barrier functions for constrained control of linear systems with input delay. In 2018 annual American control conference (ACC), pp. 3316-3321. IEEE, 2018.  
Zachary Jarvis-Wloszek, Ryan Feeley, Weehong Tan, Kunpeng Sun, and Andrew Packard. Some controls applications of sum of squares programming. In 42nd IEEE International Conference on Decision and Control (IEEE Cat. No. 03CH37475), volume 5, pp. 4676-4681. IEEE, 2003.  
Ioannis Karatzas and Steven Shreve. Brownian motion and stochastic calculus, volume 113. Springer Science & Business Media, 2012.  
S Mohammad Khansari-Zadeh and Aude Billard. Learning control lyapunov function to ensure stability of dynamical system-based robot reaching motions. Robotics and Autonomous Systems, 62(6):752-765, 2014.  
Mohammad Javad Khojasteh, Vikas Dhiman, Massimo Franceschetti, and Nikolay Atanasov. Probabilistic safety constraints for learned high relative degree system dynamics. In Learning for Dynamics and Control, pp. 781-792. PMLR, 2020.  
J Zico Kolter and Gaurav Manek. Learning stable deep dynamics models. Advances in Neural Information Processing Systems, 32:11128-11136, 2019.  
Joseph P LaSalle. Stability theory for ordinary differential equations. Journal of Differential equations, 4(1):57-65, 1968.  
Among Li, Sean P Cornelius, Y-Y Liu, Long Wang, and A-L Barabasi. The fundamental advantages of temporal networks. Science, 358(6366):1042-1046, 2017.  
Wei Lin and Yangbo He. Complete synchronization of the noise-perturbed chua's circuits. Chaos: An Interdisciplinary Journal of Nonlinear Science, 15(2):023705, 2005.  
Liang Liu, Shen Yin, Lixian Zhang, Xunyuan Yin, and Huaicheng Yan. Improved results on asymptotic stabilization for stochastic nonlinear time-delay systems with application to a chemical reactor system. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 47(1):195-204, 2016.  
X Mao. Robustness of exponential stability of stochastic differential delay equations. IEEE Transactions on Automatic Control, 41(3):442-447, 1996.  
Xuerong Mao. Lasalle-type theorems for stochastic differential delay equations. Journal of mathematical analysis and applications, 236(2):350-369, 1999.  
Xuerong Mao. A note on the lasalle-type theorems for stochastic differential delay equations. Journal of mathematical analysis and applications, 268(1):125-142, 2002.  
Xuerong Mao. Stochastic differential equations and applications. Elsevier, 2007.  
Takashi Matsumoto. A chaotic attractor from chua's circuit. IEEE Transactions on Circuits and Systems, 31(12):1055-1058, 1984.  
Pablo A Parrilo. Structured Semidefinite Programs and Semialgebraic Geometry Methods in Robustness and Optimization. California Institute of Technology, 2000.  
Dongxue Peng, Xiaodi Li, R Rakkiyappan, and Yanhui Ding. Stabilization of stochastic delayed systems: Event-triggered impulsive control. Applied Mathematics and Computation, 401:126054, 2021.

Stephen Prajna and Ali Jabbabaie. Safety verification of hybrid systems using barrier certificates. In International Workshop on Hybrid Systems: Computation and Control, pp. 477-492. Springer, 2004.  
Stephen Prajna, Ali Jabbabaie, and George J Pappas. Stochastic safety verification using barrier certificates. In 2004 43rd IEEE conference on decision and control (CDC)(IEEE Cat. No. 04CH37601), volume 1, pp. 929-934. IEEE, 2004.  
Rajesh Rajamani. Vehicle dynamics and control. Springer Science & Business Media, 2011.  
Hadi Ravanbakhsh and Sriram Sankaranarayanan. Learning control lyapunov functions from counterexamples and demonstrations. Autonomous Robots, 43(2):275-307, 2019.  
Alexander Robey, Haimin Hu, Lars Lindemann, Hanwen Zhang, Dimos V Dimarogonas, Stephen Tu, and Nikolai Matni. Learning control barrier functions from expert demonstrations. In 2020 59th IEEE Conference on Decision and Control (CDC), pp. 3717-3724. IEEE, 2020.  
Cesar Santoyo, Maxence Dutreix, and Samuel Coogan. A barrier function approach to finite-time stochastic system verification and control. Automatica, 125:109439, 2021.  
Meenakshi Sarkar, Debasish Ghose, and Evangelos A Theodorou. High-relative degree stochastic control lyapunov and barrier functions. arXiv preprint arXiv:2004.03856, 2020.  
ANShiryaev.Theoryofmartingales,1989.  
Bo Song, Ju H Park, Zheng-Guang Wu, and Xuchao Li. New results on delay-dependent stability analysis and stabilization for stochastic time-delay systems. International Journal of Robust and Nonlinear Control, 24(16):2546-2559, 2014.  
Yong-Zheng Sun, Si-Yang Leng, Ying-Cheng Lai, Celso Grebogi, and Wei Lin. Closed-loop control of complex networks: A trade-off between time and energy. Physical Review Letters, 119(19): 198301, 2017.  
Yonghui Sun and Jinde Cao. Adaptive synchronization between two different noise-perturbed chaotic systems with fully unknown parameters. Physica A: statistical mechanics and its applications, 376:253-265, 2007.  
Andrew Taylor, Andrew Singletary, Yisong Yue, and Aaron Ames. Learning for safety-critical control with control barrier functions. In Learning for Dynamics and Control, pp. 708-717. PMLR, 2020.  
Nolan Wagener, Ching-An Cheng, Jacob Sacks, and Byron Boots. An online learning approach to model predictive control. arXiv preprint arXiv:1902.08967, 2019.  
Li Wang, Aaron D Ames, and Magnus Egerstedt. Multi-objective compositions for collision-free connectivity maintenance in teams of mobile robots. In 2016 IEEE 55th Conference on Decision and Control (CDC), pp. 2659-2664. IEEE, 2016.  
Duncan J Watts and Steven H Strogatz. Collective dynamics of 'small-world'networks. nature, 393 (6684):440-442, 1998.  
Grady Williams, Paul Drews, Brian Goldfain, James M Rehg, and Evangelos A Theodorou. Information-theoretic model predictive control: Theory and applications to autonomous driving. IEEE Transactions on Robotics, 34(6):1603-1622, 2018.  
Yuhua Xu, Yuling Wang, Wuneng Zhou, and Jian'an Fang. Stochastic complex networks synchronize to the limit set with adaptive controller and adaptive delay. Mathematical Methods in the Applied Sciences, 37(15):2290-2296, 2014.  
Gang Yan, Jie Ren, Ying-Cheng Lai, Choy-Heng Lai, and Baowen Li. Controlling complex networks: How much energy is needed? Physical Review Letters, 108(21):218703, 2012.  
Xue Ying. An overview of overfitting and its solutions. In Journal of Physics: Conference Series, volume 1168, pp. 022022. IOP Publishing, 2019.

Wenwu Yu and Jinde Cao. Adaptive synchronization and lag synchronization of uncertain dynamical system with time delay based on parameter identification. Physica A: Statistical Mechanics and its Applications, 375(2):467-482, 2007.  
Jingdong Zhang, Qunxi Zhu, and Wei Lin. Neural stochastic control. arXiv preprint arXiv:2209.07240, 2022.  
Quanxin Zhu. Stabilization of stochastic nonlinear delay systems with exogenous disturbances and the event-triggered feedback control. IEEE Transactions on Automatic Control, 64(9):3764-3771, 2018.