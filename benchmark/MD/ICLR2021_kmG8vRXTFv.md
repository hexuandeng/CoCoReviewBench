# AUGMENTING PHYSICAL MODELS WITH DEEP NETWORKS FOR COMPLEX DYNAMICS FORECASTING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Forecasting complex dynamical phenomena in settings where only partial knowledge of their dynamics is available is a prevalent problem across various scientific fields. While purely data-driven approaches are arguably insufficient in this context, standard physical modeling based approaches tend to be over-simplistic, inducing non-negligible errors. In this work, we introduce the APHYNITY framework, a principled approach for augmenting incomplete physical dynamics described by differential equations with deep data-driven models. It consists in decomposing the dynamics into two components: a physical component accounting for the dynamics for which we have some prior knowledge, and a data-driven component accounting for errors of the physical model. The learning problem is carefully formulated such that the physical model explains as much of the data as possible, while the data-driven component only describes information that cannot be captured by the physical model, no more, no less. This not only provides the existence and uniqueness for this decomposition, but also ensures interpretability and benefits generalization. Experiments made on three important use cases, each representative of a different family of phenomena, i.e. reaction-diffusion equations, wave equations and the non-linear damped pendulum, show that APHYNITY can efficiently leverage approximate physical models to accurately forecast the evolution of the system and correctly identify relevant physical parameters.

# 1 INTRODUCTION

Modeling and forecasting complex dynamical systems is a major challenge in domains such as environment and climate (Rolnick et al., 2019), health science (Choi et al., 2016), and in many industrial applications (Toubeau et al., 2018). Model Based (MB) approaches typically rely on partial or ordinary differential equations (PDE/ODE) and stem from a deep understanding of the underlying physical phenomena. Machine learning (ML) and deep learning methods are more prior agnostic yet have become state-of-the-art for several spatio-temporal prediction tasks (Shi et al., 2015; Wang et al., 2018; Oreshkin et al., 2020), and connections have been drawn between deep architectures and numerical ODE solvers, e.g. neural ODEs (Chen et al., 2018; Ayed et al., 2019b). However, modeling complex physical dynamics is still beyond the scope of pure ML methods, which often cannot properly extrapolate to new conditions as MB approaches do.

Combining the MB and ML paradigms is an emerging trend to develop the interplay between the two paradigms. For example, Brunton et al. (2016); Long et al. (2018b) learn the explicit form of PDEs directly from data, Raissi et al. (2019); Sirignano & Spiliopoulos (2018) use NNs as implicit methods for solving PDEs, Seo et al. (2020) learn spatial differences with a graph network, Ummenhofer et al. (2020) introduce continuous convolutions for fluid simulations, de Bézenac et al. (2018) learn the velocity field of an advection-diffusion system, Greydanus et al. (2019); Chen et al. (2020) enforce conservation laws in the network architecture or in the loss function.

The large majority of aforementioned MB/ML hybrid approaches assume that the physical model adequately describes the observed dynamics. This assumption is, however, commonly violated in practice. This may be due to various factors, e.g. idealized assumptions and difficulty to explain processes from first principles (Gentine et al., 2018), computational constraints prescribing a fine grain modeling of the system (Ayed et al., 2019a), unknown external factors, forces and sources which are present (Large & Yeager, 2004). In this paper, we aim at leveraging prior dynamical

![](images/bb7d302a3f4cf82838d3a12728c79f564ba78d7db6ac0bbd4fa153876e27a711.jpg)  
(a) Data-driven Neural ODE

![](images/cb27027f155bb126993443196e2a6e85e8c0d0fdff566cc88ba8b0a9b0eca215.jpg)  
Figure 1: Predicted dynamics for the damped pendulum vs. ground truth (GT) trajectories  $\mathrm{d}^2\theta/\mathrm{dt}^2 + \omega_0^2\sin\theta + \alpha\mathrm{d}\theta/\mathrm{dt} = 0$ . We show that in (a) the data-driven approach (Chen et al., 2018) fails to properly learn the dynamics due to the lack of training data, while in (b) an ideal pendulum cannot take friction into account. The proposed APHYNITY shown in (c) augments the over-simplified physical model in (b) with a data-driven component. APHYNITY improves both forecasting (MSE) and parameter identification (Error  $T_0$ ) compared to (b).  
(b) Simple physical model

![](images/58eb8d24ab4d7ce4003ccba79a4f2afec74436330a497c1a6224c73b97ac3ded.jpg)  
(c) Our APHYNITY framework

ODE/PDE knowledge in situations where this physical model is incomplete, i.e. unable to represent the whole complexity of observed data. To handle this case, we introduce a principled learning framework to Augment incomplete PHYSical models for ideNtIfying and forecastTing complex dYnamics (APHYNITY). The rationale of APHYNITY, illustrated in Figure 1 on the pendulum problem, is to augment the physical model when—and only when—it falls short.

Designing a general method for combining MB and ML approaches is still a widely open problem, and a clear problem formulation for the latter is lacking (Reichstein et al., 2019). Our contributions towards these goals are the following:

- We introduce a simple yet principled framework for combining both approaches. We decompose the data into a physical and a data-driven term such that the data-driven component only models information that cannot be captured by the physical model. We provide existence and uniqueness guarantees (Section 3.1) for the decomposition given mild conditions, and show that this formulation ensures interpretability and benefits generalization.  
- We propose a trajectory-based training formulation (Section 3.2) along with an adaptive optimization scheme (Section 3.3) enabling end-to-end learning for both physical and deep learning components. This allows APHYNITY to automatically adjust the complexity of the neural network to different approximation levels of the physical model, paving the way to learning flexible hybrid models.  
- We demonstrate the generality of the approach on three use cases (reaction-diffusion, wave equations and the pendulum) representative of different PDE families (parabolic, hyperbolic), having a wide spectrum of application domains, e.g. acoustics, electromagnetism, chemistry, biology, physics (Section 4). We show that APHYNITY can perform similarly to complete physical models by augmenting incomplete ones, both in forecasting accuracy and physical parameter identification. APHYNITY can also be successfully extended to the partially observable setting (see discussion in Section 5).

# 2 RELATED WORK

Correction in data assimilation Prediction under approximate physical models has been tackled by traditional statistical calibration techniques, which often rely on bayesian methods (see the review Pernot & Cailliez, 2017). In data assimilation techniques, e.g. the Kalman filter (Kalman, 1960), 4Dvar (Courtier et al., 1994), prediction errors and observation processes are modeled probabilistically and an optimal correction with observed data is applied after each prediction step. Similarly in model-based reinforcement learning, model deficiencies are typically handled by considering only short-term rollouts (Janner et al., 2019) or by model predictive control (Becker et al., 2019). The main originality of APHYNITY is to learn the correction term with deep NNs, and to design a decomposition enabling the physical and correction components to cooperate during training.

Augmented physical models Recently, some attempts have been made to explore the problem of cooperation between physical and data-driven models. HybridNet (Long et al., 2018a) and PhICNet (Saha et al., 2020) both use data-driven networks to learn additive perturbations or source terms to a given PDE. The former considers the favorable context where the perturbations can be accessed, and the latter the special case of additive noise on the input. Wang et al. (2019a); Mehta et al. (2020) propose several empirical fusion strategies with deep nets but they are not grounded on theoretical arguments. PhyDNet (Le Guen & Thome, 2020) tackles augmentation in partially-observed settings, but with specific recurrent architectures dedicated to video prediction. Crucially, all the aforementioned approaches do not address the issues of uniqueness of the decomposition or of proper cooperation for correct parameter identification. Besides, we found experimentally that this vanilla cooperation is inferior to the APHYNITY learning scheme in terms of forecasting and parameter identification performances (see experiments in Section 4.2).

# 3 THE APHYNITY MODEL

In the following, we study dynamics driven by an equation of the form:

$$
\frac {\mathrm {d} X _ {t}}{\mathrm {d} t} = F \left(X _ {t}\right) \tag {1}
$$

defined over a finite time interval  $[0,T]$ , where the state  $X$  is either vector-valued, i.e. we have  $X_{t}\in \mathbb{R}^{d}$  for every  $t$ , (pendulum equations in Section 4), or  $X_{t}$  is a  $d$ -dimensional vector field over a spatial domain  $\Omega \subset \mathbb{R}^k$ , with  $k\in \{2,3\}$ , i.e.  $X_{t}(x)\in \mathbb{R}^{d}$  for every  $(t,x)\in [0,T]\times \Omega$  (reaction-diffusion and wave equations in Section 4). We suppose that we have access to a set of observed trajectories  $\mathcal{D} = \{X.:[0,T]\to \mathcal{A}\mid \forall t\in [0,T],\mathrm{d}X_{t} / \mathrm{d}t = F(X_{t})\}$ , where  $\mathcal{A}$  is the set of  $X$  values (either  $\mathbb{R}^d$  or vector field). In our case,  $F:\mathcal{A}\rightarrow \mathcal{A}$  is unknown and we only assume that  $F\in \mathcal{F}$ , with  $(\mathcal{F},\parallel \cdot \parallel)$  a normed vector space.

# 3.1 DECOMPOSING DYNAMICS INTO PHYSICAL AND AUGMENTED TERMS

As introduced in Section 1, we consider the common situation where incomplete information is available on the dynamics, under the form of a family of ODEs or PDEs characterized by their temporal evolution  $\mathcal{F}_p\subset \mathcal{F}$ . The APHYNITY framework leverages the knowledge of  $\mathcal{F}_p$  while mitigating the approximations induced by this simplified model through the combination of physical and data-driven components.  $\mathcal{F}$  being a vector space, we can write:

$$
F = F _ {p} + F _ {a}
$$

where  $F_{p} \in \mathcal{F}_{p}$  encodes the incomplete physical knowledge and  $F_{a} \in \mathcal{F}$  is the data-driven augmentation term complementing  $F_{p}$ . The incomplete physical prior is supposed to belong to a known family, but the physical parameters (e.g. propagation speed for the wave equation) are unknown and need to be estimated from data. Both  $F_{p}$  and  $F_{a}$  parameters are estimated by fitting the trajectories from  $\mathcal{D}$ .

The decomposition  $F = F_{p} + F_{a}$  is in general not unique. For example, all the dynamics could be captured by the  $F_{a}$  component. This decomposition is thus ill-defined, which hampers the interpretability and the extrapolation abilities of the model. In other words, one wants the estimated parameters of  $F_{p}$  to be as close as possible to the true parameter values of the physical model and  $F_{a}$  to play only a complementary role w.r.t  $F_{p}$ , so as to model only the information that cannot be captured by the physical prior. For example, when  $F \in \mathcal{F}_{p}$ , the data can be fully described by the physical model, and in this case it is sensible to desire  $F_{a}$  to be nullified; this is of central importance in a setting where one wishes to identify physical quantities, and for the model to generalize and extrapolate to new conditions. In a more general setting where the physical model is incomplete, the action of  $F_{a}$  on the dynamics, as measured through its norm, should be as small as possible.

This general idea is embedded in the following optimization problem:

$$
\min  _ {F _ {p} \in \mathcal {F} _ {p}, F _ {a} \in \mathcal {F}} \| F _ {a} \| \quad \text {s u b j e c t t o} \quad \forall X \in \mathcal {D}, \forall t, \frac {\mathrm {d} X _ {t}}{\mathrm {d} t} = \left(F _ {p} + F _ {a}\right) \left(X _ {t}\right) \tag {2}
$$

A first key question is whether the minimum in Eq. (2) is indeed well-defined, in other words whether there exists indeed a decomposition with a minimal norm  $F_{a}$ . The answer actually depends on the geometry of  $\mathcal{F}_p$ , and is formulated in the following proposition proven in Appendix B:

Proposition 1 (Existence of a minimizing pair). If  $\mathcal{F}_p$  is a proximal set $^1$ , there exists a decomposition minimizing Eq. (2).

Proximinality is a mild condition which, as shown through the proof of the proposition, cannot be weakened. It is a property verified by any boundedly compact set. In particular, it is true for closed subsets of finite dimensional spaces. However, if only existence is guaranteed, while forecasts would be expected to be accurate, non-uniqueness of the decomposition would hamper the interpretability of  $F_{p}$  and this would mean that the identified physical parameters are not uniquely determined.

It is then natural to ask under which conditions solving problem Eq. (2) leads to a unique decomposition into a physical and a data-driven component. The following result provides guarantees on the existence and uniqueness of the decomposition under mild conditions. The proof is given in Appendix B:

Proposition 2 (Uniqueness of the minimizing pair). If  $\mathcal{F}_p$  is a Chebyshev set $^1$ , Eq. (2) admits a unique minimizer. The  $F_p$  in this minimizer pair is the metric projection of the unknown  $F$  onto  $\mathcal{F}_p$ .

The Chebyshev assumption condition is strictly stronger than proximinality but si still quite mild and necessary. Indeed, in practice, many sets of interest are Chebyshev, including all closed convex spaces in strict normed spaces and, if  $\mathcal{F} = L^2$ ,  $\mathcal{F}_p$  can be any closed convex set, including all finite dimensional subspaces. In particular, all examples considered in the experiments are Chebyshev sets.

Propositions 1 and 2 provide, under mild conditions, the theoretical guarantees for the APHYNITY formulation to infer the correct MB/ML decomposition, thus enabling both recovering the proper physical parameters and accurate forecasting.

# 3.2 SOLVING APHYNITY WITH DEEP NEURAL NETWORKS

In the following, both terms of the decomposition are parametrized and are denoted as  $F_{p}^{\theta_{p}}$  and  $F_{p}^{\theta_{a}}$ . Solving APHYNITY then consists in estimating the parameters  $\theta_{p}$  and  $\theta_{a}$ .  $\theta_{p}$  are the physical parameters and are typically low-dimensional, e.g. 2 or 3 in our experiments for the considered physical models. For  $F_{a}$ , we need sufficiently expressive models able to optimize over all  $\mathcal{F}$ : we thus use deep neural networks, which have shown promising performances for the approximation of differential equations (Raissi et al., 2019; Ayed et al., 2019b).

When learning the parameters of  $F_{p}^{\theta_{p}}$  and  $F_{a}^{\theta_{a}}$ , we have access to a finite dataset of trajectories discretized with a given temporal resolution  $\Delta t$ :  $\mathcal{D}_{\mathrm{train}} = \{(X_{k\Delta t}^{(i)})_{0 \leq k \leq \lfloor T / \Delta t \rfloor}\}_{1 \leq i \leq N}$ . Solving Eq. (2) requires estimating the state derivative  $\frac{\mathrm{d}X_{t}}{\mathrm{d}t}$  appearing in the constraint term. One solution is to approximate this derivative using e.g. finite differences as in (Brunton et al., 2016; Greydanus et al., 2019; Cranmer et al., 2020). This numerical scheme requires high space and time resolutions in the observation space in order to get reliable gradient estimates. Furthermore it is often unstable, leading to explosive numerical errors as discussed in Appendix D. We propose instead to solve Eq. (2) using an integral trajectory-based approach: we compute  $\widetilde{X}_{k\Delta t,X_0}^i$  from an initial state  $X_0^{(i)}$  using the current  $F_{p}^{\theta_{p}} + F_{a}^{\theta_{a}}$  dynamics, then enforce the constraint  $\widetilde{X}_{k\Delta t,X_0}^i = X_{k\Delta t}^i$ . This leads to our final objective function on  $(\theta_p,\theta_a)$ :

$$
\min  _ {\theta_ {p}, \theta_ {a}} \| F _ {a} ^ {\theta_ {a}} \| \quad \text {s u b j e c t} \quad \forall i, \forall k, \widetilde {X} _ {k \Delta t} ^ {(i)} = X _ {k \Delta t} ^ {(i)} \tag {3}
$$

where  $\widetilde{X}_{k\Delta t}^{(i)}$  is the approximate solution of the integral  $\int_{X_0^{(i)}}^{X_0^{(i)} + k\Delta t}(F_p^{\theta_p} + F_a^{\theta_a})(X_s)\mathrm{d}X_s$  obtained by a differentiable ODE solver.

In our setting, where we consider situations for which  $F_{p}^{\theta_{p}}$  only partially describes the physical phenomenon, this coupled MB + ML formulation leads to different parameter estimates than using the MB formulation alone, as analyzed more thoroughly in Appendix C. Interestingly, our experiments show that using this formulation also leads to a better identification of the physical parameters  $\theta_{p}$  than when fitting the simplified physical model  $F_{p}^{\theta_{p}}$  alone (Section 4). With only an incomplete knowledge

on the physics,  $\theta_p$  estimator will be biased by the additional dynamics which needs to be fitted in the data. Appendix F also confirms that the integral formulation gives better forecasting results and a more stable behavior than supervising over finite difference approximations of the derivatives.

# 3.3 ADAPTIVELY CONSTRAINED OPTIMIZATION

The formulation in Eq. (3) involves constraints which are difficult to enforce exactly in practice. We considered a variant of the method of multipliers (Bertsekas, 1996) which uses a sequence of Lagrangian relaxations  $\mathcal{L}_{\lambda_j}(\theta_p,\theta_a)$ :

$$
\mathcal {L} _ {\lambda_ {j}} \left(\theta_ {p}, \theta_ {a}\right) = \left\| F _ {a} ^ {\theta_ {a}} \right\| + \lambda_ {j} \cdot \mathcal {L} _ {t r a j} \left(\theta_ {p}, \theta_ {a}\right) \tag {4}
$$

where  $\mathcal{L}_{\text{traj}}(\theta_p, \theta_a) = \sum_{i=1}^{N} \sum_{h=1}^{T/\Delta t} \|X_{h\Delta t}^{(i)} - \widetilde{X}_{h\Delta t}^{(i)}\|$ .

This method needs an increasing sequence  $(\lambda_j)_j$  such that the successive minima of  $\mathcal{L}_{\lambda_j}$  converge to a solution (at least a local one) of the constrained problem Eq. (3). We select  $(\lambda_j)_j$  by using an iterative strategy: starting from a value  $\lambda_0$ , we iterate, minimizing  $\mathcal{L}_{\lambda_j}$  by gradient descent, then update  $\lambda_j$  with  $\lambda_{j + 1} = \lambda_j + \tau_2\mathcal{L}_{traj}(\theta_{j + 1})$ , where  $\tau_{2}$  is a chosen hyper-parameter and  $\theta = (\theta_p,\theta_a)$ . This procedure is summarized in Algorithm 1. This adaptive iterative procedure allows us to obtain stable and robust results, in a reproducible fashion, as shown in the experiments.

# Algorithm 1: APHYNITY

Initialization:  $\lambda_0\geq 0,\tau_1 > 0,\tau_2 > 0;$  for epoch  $= 1:N_{\text{epochs}}$  do

$$
\begin{array}{c} \text {f o r i t e r i n 1 : N _ {i t e r} d o} \\ \left| \begin{array}{c} \text {f o r b a t c h i n 1 : B d o} \\ \theta_ {j + 1} = \theta_ {j} - \\ \tau_ {1} \nabla \left[ \lambda_ {j} \mathcal {L} _ {t r a j} (\theta_ {j}) + \| F _ {a} \| \right] \\ \lambda_ {j + 1} = \lambda_ {j} + \tau_ {2} \mathcal {L} _ {t r a j} (\theta_ {j + 1}) \end{array} \right| \\ \end{array}
$$

# 4 EXPERIMENTAL VALIDATION

We validate our approach on 3 classes of challenging physical dynamics: reaction-diffusion, wave propagation, and the damped pendulum, representative of various application domains such as chemistry, biology or ecology (for reaction-diffusion) and earth physi, acoustic, electromagnetism or even neuro-biology (for waves equations). The two first dynamics are described by PDEs and thus in practice should be learned from very high-dimensional vectors, discretized from the original compact domain. This makes the learning much more difficult than from the one-dimensional pendulum case. For each problem, we investigate the cooperation between physical models of increasing complexity encoding incomplete knowledge of the dynamics (denoted Incomplete physics in the following) and data-driven models. We show the relevance of APHYNITY (denoted APHYNITY models) both in terms of forecasting accuracy and physical parameter identification.

# 4.1 EXPERIMENTAL SETTING

We describe the three families of equations studied in the experiments. In all experiments,  $\mathcal{F} = \mathcal{L}^2 (\mathcal{A})$  where  $\mathcal{A}$  is the set of all admissible states for each problem, and the  $\mathcal{L}^2$  norm is computed on  $\mathcal{D}_{train}$  by:  $\| F\| ^2\approx \sum_{i,k}\| F(X_{k\Delta t}^{(i)})\| ^2$ . All considered sets of physical functionals  $\mathcal{F}_p$  are closed and convex in  $\mathcal{F}$  and thus are Chebyshev. In order to enable the evaluation on both prediction and parameter identification, all our experiments are conducted on simulated datasets with known model parameters. Each dataset has been simulated using an appropriate high-precision integration scheme for the corresponding equation. All solver-based models take the first state  $X_0$  as input and predict the remaining time-steps by integrating  $F$  through the same differentiable generic and common ODE solver (4th order Runge-Kutta) $^3$ . Implementation details and architectures are given in Appendix E.

Reaction-diffusion equations We consider a FitzHugh-Nagumo type model (Klaasen & Troy, 1984), driven by the PDE  $\partial u / \partial t = a\Delta u + R_u(u,v;k),\partial v / \partial t = b\Delta v + R_v(u,v)$  where  $a,b$  are the diffusion coefficients,  $\Delta$  is the Laplace operator.  $R_{u}$  and  $R_{v}$  are local reaction terms. The state is  $X = (u,v)$  and is defined over a compact spatial domain  $\Omega$  with periodic boundary

conditions. The considered physical models are:  $\bullet$  Param PDE  $(a,b)$ , without reaction terms:  $\mathcal{F}_p = \{F_p^{a,b}:(u,v)\mapsto (a\Delta u,b\Delta v)\mid a,b\in [\epsilon , + \infty)$  with  $\epsilon >0\}$ $\bullet$  Param PDE  $(a,b,k)$ :  $\mathcal{F}_p = \{F_p^{a,b,k}:(u,v)\mapsto (a\Delta u + R_u(u,v;k),b\Delta v + R_v(u,v))\mid a,b,k\in [\epsilon , + \infty)$  with  $\epsilon >0\}$ .

Damped wave equations We investigate the damped-wave PDE:  $\partial^2 w / \partial t^2 -c^2\Delta w + k\partial w / \partial t = 0$  where  $k$  is the damping coefficient. The state is  $X = (w,\partial w / \partial t)$  and we consider a compact spatial domain  $\Omega$  with Neumann homogeneous boundary conditions. Note that this damping differs from the pendulum, as its effect is global. Our physical models are:  $\bullet$  Param PDE  $(c)$ , without damping term:  $\mathcal{F}_p = \{F_p^c:(u,v)\mapsto (v,c^2\Delta u)\mid c\in [\epsilon , + \infty)$  with  $\epsilon >0\}$ ;  $\bullet$  Param PDE  $(c,k)$ :  $\mathcal{F}_p = \{F_p^{c,k}:(u,v)\mapsto (v,c^2\Delta u - kv)\mid c,k\in [\epsilon , + \infty)$  with  $\epsilon >0\}$ .

Damped pendulum The evolution follows the ODE  $\mathrm{d}^2\theta/\mathrm{d}t^2 + \omega_0^2\sin\theta + \alpha\mathrm{d}\theta/\mathrm{d}t = 0$ , where  $\theta(t)$  is the angle,  $\omega_0$  the proper pulsation ( $T_0$  the period) and  $\alpha$  the damping coefficient. With state  $X = (\theta, \mathrm{d}\theta/\mathrm{d}t)$ , the ODE is  $F_p^{\omega_0,\alpha}: X \mapsto (\mathrm{d}\theta/\mathrm{d}t, -\omega_0^2\sin\theta - \alpha\mathrm{d}\theta/\mathrm{d}t)$ . Our physical models are: Hamiltonian (Greydanus et al., 2019), a conservative approximation, with  $\mathcal{F}_p = \{F_p^\mathcal{H}:(u,v) \mapsto (\partial_y\mathcal{H}(u,v), -\partial_x\mathcal{H}(u,v)) \mid \mathcal{H} \in H^1(\mathbb{R}^2)\}$ ,  $H^1(\mathbb{R}^2)$  is the first order Sobolev space. Param ODE  $(\omega_0)$ , the frictionless pendulum:  $\mathcal{F}_p = \{F_p^{\omega_0,\alpha=0} \mid \omega_0 \in [\epsilon, +\infty)$  with  $\epsilon > 0$ . Param ODE  $(\omega_0,\alpha)$ , the full pendulum equation:  $\mathcal{F}_p = \{F_p^{\omega_0,\alpha} \mid \omega_0,\alpha \in [\epsilon, +\infty)$  with  $\epsilon > 0$ .

Baselines As purely data-driven baselines, we use Neural ODE (Chen et al., 2018) for the three problems and PredRNN++ (Wang et al., 2018, for reaction-diffusion only) which are competitive models for datasets generated by differential equations and for spatio-temporal data. As MB/ML methods, in the ablations studies (see Appendix F), we compare for all problems, to the vanilla MB/ML cooperation scheme found in (Wang et al., 2019a; Mehta et al., 2020). We also show results for True PDE/ODE, which corresponds to the equation for data simulation (which do not lead to zero error due to the difference between simulation and training integration schemes). For the pendulum, we compare to Hamiltonian neural networks (Greydanus et al., 2019; Toth et al., 2020) and to the deep Galerkin method (DGM, Sirignano & Spiliopoulos, 2018). See additional details in Appendix E.

# 4.2 RESULTS

We analyze and discuss below the results obtained for the three kind of dynamics. We successively examine different evaluation or quality criteria. The conclusions are consistent for the three problems, which allows us to highlight clear trends for all of them.

Forecasting accuracy The data-driven models do not perform well compared to True PDE/ODE (all values in log MSE): -4.6 for PredRNN++ vs. -9.17 for reaction-diffusion, -2.51 vs. -5.24 for wave equation, and -2.84 vs. -8.44 for the pendulum in Table 1. The Deep Galerkin method for the pendulum in complete physics  $DGM(\omega_0, \alpha)$ , being constrained by the equation, outperforms Neural ODE but is far inferior to APHYNITY models. In the incomplete physics case,  $DGM(\omega_0)$  fails to compensate for the missing information. The incomplete physical models, Param PDE ( $a, b$ ) for the reaction-diffusion, Param PDE ( $c$ ) for the wave equation, and Param ODE ( $\omega_0$ ) and Hamiltonian models for the damped pendulum, have even poorer performances than purely data-driven ones, as can be expected since they ignore important dynamical components, e.g. friction in the pendulum case. Using APHYNITY with these imperfect physical models greatly improves forecasting accuracy in all cases, significantly outperforming purely data-driven models, and reaching results often close to the accuracy of the true ODE, when APHYNITY and the true ODE models are integrated with the same numerical scheme (which is different from the one used for data generation, hence the non-null errors even for the true equations), e.g. -5.92 vs. -5.24 for wave equation in Table 1. This clearly highlights the capacity of our approach to augment incomplete physical models with a learned data-driven component.

Physical parameter estimation Confirming the phenomenon mentioned in the introduction and detailed in Appendix C, incomplete physical models can lead to bad estimates for the relevant physical parameters: an error respectively up to  $67.6\%$  and  $10.4\%$  for parameters in the reaction-diffusion and wave equations, and an error of more than  $13\%$  for parameters for the pendulum in Table 1. APHYNITY is able to significantly improve physical parameters identification:  $2.3\%$  error for the reaction-diffusion,  $0.3\%$  for the wave equation, and  $4\%$  for the pendulum. This validates the fact that augmenting a simple physical model to compensate its approximations is not only beneficial for

Table 1: Forecasting and identification results on the (a) reaction-diffusion, (b) wave equation, and (c) damped pendulum datasets. We set for (a)  $a = 1 \times 10^{-3}$ ,  $b = 5 \times 10^{-3}$ ,  $k = 5 \times 10^{-3}$ , for (b)  $c = 330$ ,  $k = 50$  and for (c)  $T_0 = 6$ ,  $\alpha = 0.2$  as true parameters. log MSEs are computed respectively over 25, 25, and 40 predicted time-steps. %Err param. averages the results when several physical parameters are present. For each level of incorporated physical knowledge, equivalent best results according to a Student t-test are shown in bold. n/a corresponds to non-applicable cases.  

<table><tr><td colspan="2">Dataset</td><td>Method</td><td>log MSE</td><td>%Err param.</td><td>\( \left\| F_{a}\right\|^2 \)</td></tr><tr><td rowspan="8">(a)Reaction-diffusion</td><td rowspan="2">Data-driven</td><td>Neural ODE</td><td>-3.76±0.02</td><td>n/a</td><td>n/a</td></tr><tr><td>PredRNN++</td><td>-4.60±0.01</td><td>n/a</td><td>n/a</td></tr><tr><td rowspan="2">Incomplete physics</td><td>Param PDE (a,b)</td><td>-1.26±0.02</td><td>67.6</td><td>n/a</td></tr><tr><td>APHYNITY Param PDE (a,b)</td><td>-5.10±0.21</td><td>2.3</td><td>67</td></tr><tr><td rowspan="4">Complete physics</td><td>Param PDE (a,b,k)</td><td>-9.34±0.20</td><td>0.17</td><td>n/a</td></tr><tr><td>APHYNITY Param PDE (a,b,k)</td><td>-9.35±0.02</td><td>0.096</td><td>1.5e-6</td></tr><tr><td>True PDE</td><td>-8.81±0.05</td><td>n/a</td><td>n/a</td></tr><tr><td>APHYNITY True PDE</td><td>-9.17±0.02</td><td>n/a</td><td>1.4e-7</td></tr><tr><td rowspan="7">(b)Wave equation</td><td>Data-driven</td><td>Neural ODE</td><td>-2.51±0.29</td><td>n/a</td><td>n/a</td></tr><tr><td rowspan="2">Incomplete physics</td><td>Param PDE (c)</td><td>0.51±0.07</td><td>10.4</td><td>n/a</td></tr><tr><td>APHYNITY Param PDE (c)</td><td>-4.64±0.25</td><td>0.3</td><td>71.</td></tr><tr><td rowspan="4">Complete physics</td><td>Param PDE (c,k)</td><td>-4.68±0.55</td><td>1.38</td><td>n/a</td></tr><tr><td>APHYNITY Param PDE (c,k)</td><td>-6.09±0.28</td><td>0.63</td><td>4.35</td></tr><tr><td>True PDE</td><td>-4.66±0.30</td><td>n/a</td><td>n/a</td></tr><tr><td>APHYNITY True PDE</td><td>-5.24±0.45</td><td>n/a</td><td>0.14</td></tr><tr><td rowspan="11">(c)Damped pendulum</td><td>Data-driven</td><td>Neural ODE</td><td>-2.84±0.70</td><td>n/a</td><td>n/a</td></tr><tr><td rowspan="5">Incomplete physics</td><td>Hamiltonian</td><td>-0.35±0.10</td><td>n/a</td><td>n/a</td></tr><tr><td>APHYNITY Hamiltonian</td><td>-3.97±1.20</td><td>n/a</td><td>623</td></tr><tr><td>Param ODE (ω0)</td><td>-0.14±0.10</td><td>13.2</td><td>n/a</td></tr><tr><td>Deep Galerkin Method (ω0)</td><td>-3.10±0.40</td><td>22.1</td><td>n/a</td></tr><tr><td>APHYNITY Param ODE (ω0)</td><td>-7.86±0.60</td><td>4.0</td><td>132</td></tr><tr><td rowspan="5">Complete physics</td><td>Param ODE (ω0,α)</td><td>-8.28±0.40</td><td>0.45</td><td>n/a</td></tr><tr><td>Deep Galerkin Method (ω0,α)</td><td>-3.14±0.40</td><td>7.1</td><td>n/a</td></tr><tr><td>APHYNITY Param ODE (ω0,α)</td><td>-8.31±0.30</td><td>0.39</td><td>8.5</td></tr><tr><td>True ODE</td><td>-8.58±0.20</td><td>n/a</td><td>n/a</td></tr><tr><td>APHYNITY True ODE</td><td>-8.44±0.20</td><td>n/a</td><td>2.3</td></tr></table>

prediction, but also helps to limit errors for parameter identification when dynamical models do not fit data well. This is crucial for interpretability and explainability of the estimates.

Ablation study We conduct ablation studies to validate the importance of the APHYNITY augmentation compared to a naive strategy consisting in learning  $F = F_{p} + F_{a}$  without taking care on the quality of the decomposition, as done in (Wang et al., 2019a; Mehta et al., 2020). Results shown in Table 1 of Appendix F show a consistent gain of APHYNITY for the three use cases and for all physical models: for instance for Param ODE  $(a,b)$  in reaction-diffusion, both forecasting performances  $(\log \mathrm{MSE} = -5.10$  vs.  $-4.56)$  and identification parameter  $(\mathrm{Error} = 2.33\%$  vs.  $6.39\%$  ) improve. Other ablation results are provided in Appendix F showing the relevance of the the trajectory-based approach described in Section 3.2 (vs supervising over finite difference approximations of the derivative  $F$ ).

Flexibility When applied to complete physical models, APHYNITY does not degrade accuracy, contrary to a vanilla cooperation scheme (see ablations in Appendix F). This is due to the least action principle of our approach: when the physical knowledge is sufficient for properly predicting the observed dynamics, the model learns to ignore the data-driven augmentation. This is shown by the norm of the trained neural net component  $F_{a}$ , which is reported in Table 1 last column: as expected,  $\| F_{a} \|^{2}$  diminishes as the complexity of the corresponding physical model increases, and becomes almost null for complete physical models. We see that the norm of  $F_{a}$  is a good indication of how imperfect the physical models  $\mathcal{F}_{p}$  are. It highlights the flexibility of APHYNITY to successfully adapt to very different levels of prior knowledge. Note also that APHYNITY sometimes slightly improves over the true ODE, because it compensates the error introduced by different numerical integration methods for data simulation and training (see Appendix E).

![](images/377cc747f358436fbf6231bab5f5e27ba5cb2b7911c6d26a29a6518ec329687d.jpg)  
Figure 2: Comparison of predictions of two components  $u$  (top) and  $v$  (bottom) of the reaction-diffusion system. Note that  $t = 4$  is largely beyond the dataset horizon ( $t = 2.5$ ).

![](images/8c9e1b905997e3fb90c4f015bb5c03df16fde42f99e4f85f85da23eda40d567d.jpg)

![](images/898e8f5fee7645852c55b43a860e67bc8ea5140724994c9827817543350c87c0.jpg)

![](images/a5d8cb27c9218b410d597f3af5f9820c3695c53ff736d6d6ca4d4b1eee7f05e7.jpg)

![](images/350064f98d3d132677d99a07cdd1a6ef484ddfd741e4f0ae753a1b2c29e3db18.jpg)

![](images/43d681146f393cd157c39e27e5c07c59404e6d26a4df4609dcbf969ee8d9ccf5.jpg)

![](images/fb216486c5d3c57bda064830319a1ba91f1f28a6134ddc3e7f6871504c2b5e96.jpg)

![](images/41b761536c2dea81bdf0614c70e97bccce3286981dad604c2b945ff965cd60b5.jpg)

![](images/5e12db87c46d0b9a76e2be960629ed298cc51f8e87f1bbaf45d071b8d8a0e230.jpg)

![](images/0e11c1d50af1b29b4741e886c40d2e86b8d370b4380790702c18d0755a33aa6c.jpg)

![](images/ea948c2605d7c50855d224ada81ae9222ed0fccfefcffc3c2f0ba8d01b480d39.jpg)

![](images/5aad84bcdf6e6145a6be51b5b67bca02a24b1abc2cfc64c76fe14103707107f8.jpg)

![](images/095ef7a0861d7486a020ab29c6a66c56a88cdc7b517eb66a869e4847a595af16.jpg)

![](images/8532a8d710cf7f805bbd8e6469fd47099cf87d6c776c047063ed723f099c3bb2.jpg)

![](images/a4f90db83170480340bfbb70a882b38bb5fd6e1007174f505a4616c9d25d28ba.jpg)

![](images/502fb4e09dc142a47c1ff5b2d72f8d006b1de798a55cd84b37e66aa7794ddc01.jpg)  
(c) Ground truth simulation

![](images/6904f34d955fd3885c08235ef0e9f2f0be724c5c7d321003b4176b35d7c304f9.jpg)

![](images/a457ae13743d1ffdb8e46ed450b7fdd94ecf7e5a74260a33834791c1a4c4a418.jpg)

![](images/5d2db3d67162cefb968bc6d6636195e1ed64edc10b04cc80d97060750cc979b5.jpg)

![](images/61dd5c6c665bf0d5d7b34c5c69834a3b705b7968d19d579984297283b3b51349.jpg)

![](images/b65e4a5872b3b5c736ea1d4b045b80600eb9ad405d581ef19d23827a40122e77.jpg)

![](images/842c48a1e9ef9dfb6857272e9bf79def9775b5abd6c9a6b518d366a3d25bee57.jpg)  
(a) Neural ODE

![](images/7c410e80bbd1e9b969cf52bf2b84dfc028321985732c24fa2a4ca026cdf4d2f0.jpg)  
Figure 3: Comparison between the prediction of APHYNITY when  $c$  is estimated and Neural ODE for the damped wave equation. Note that  $t + 32$ , last column for (a, b, c) is already beyond the training time horizon ( $t + 25$ ), showing the consistency of APHYNITY method.

![](images/7dff2e3475705ae70e88bbd222f31f842fcbd0ce245407be49fac8974cb19b02.jpg)

![](images/1100fe67a4a0bccdd10bf3505f2ebed223b9c69d45b7a61e36db607daef17b5c.jpg)

![](images/6136e2766388bfcd044e462da6ddd8d4c084a51d6a7bf71a4e42cfe00373ac52.jpg)

![](images/646feed452f0e718e893c6cfb160632d4d26e2310d277bf455376085d88f1384.jpg)

![](images/9b2d76e104d02f37ce1601bdd5a174b64e373c5e87ec04dabfb34d94a9139515.jpg)  
(b) APHYNITY Param PDE (c)

![](images/e0f8d9125b978a38b0bd8b2d4cad7aa9a27af179468e12833988414935366326.jpg)

![](images/2df3a58ebeeae90ceb4be1d6d5009c5ac6f1c560bcea53cebbf2e69aba68fadf.jpg)

![](images/f8034d2555f0481e2aa7e8f1e7f54af3759e6734234495a3a0dc91c439e0befc.jpg)

![](images/b0c27c2b3a278f026a6c7ef0329e67fe82bfae45c4e6414b2cd1579bfe47be88.jpg)

![](images/3233daa45db0b2144fe67145c896f7e566e14ece50c8397e4234f9492993e638.jpg)

![](images/eeafb4a02c38fb2f18ddd5d93f017fd7e356e2338ca973ce25a3ce93e7254d27.jpg)

![](images/a32e925434d2970d70ba1b3f9205e3e56f75396958186fd3ecd8dfd2b8799b0a.jpg)

![](images/ed575ad7664e9ec73dfa279d9efa5ce0df59f4ff39b38750607e1ac7574d1f43.jpg)  
(c) Ground truth simulation

![](images/c072d03115934bbef7c926e2ab48ea0829d158f1d76aabd4d41ff7fb5d22006e.jpg)

![](images/b8807e6e90dd4a3f5048eb0604bd36de76d01d443edc7594abd0009d0d67cb15.jpg)

![](images/b3b6a40c6c8a7fae8db96675f4521796ddec0d486086b6fc42a3cf668f0c1715.jpg)

![](images/e337f63aca18ef94341376ba187127ad003fce693af34785baf2d102bbf43cc7.jpg)

![](images/a88392b0768b85672b9640f8a086f23cc62c1fbb7a4dde4bb0c3719802ad6638.jpg)

![](images/7bd8dde1d1de6e3d601d948a10710a55727651f9b015ebf1f3aad91fc43c8c7c.jpg)

![](images/b56c25d8d6729bd965279d3021d9bc9b56b9f001e868484e5f40deaac40b3443.jpg)

Qualitative visualizations Results in Figure 2 for reaction-diffusion show that the incomplete diffusion parametric PDE in Figure 2(a) is unable to properly match ground truth simulations: the behavior of the two components in Figure 2(a) is reduced to simple independent diffusions due to the lack of interaction terms between  $u$  and  $v$ . By using APHYNITY in Figure 2(b), the correlation between the two components appears together with the formation of Turing patterns, which is very similar to the ground truth. This confirms that  $F_{a}$  can learn the reaction terms and improve prediction quality. In Figure 3, we see for the wave equation that the data-driven Neural ODE model fails at approximating  $\frac{\mathrm{d}w}{\mathrm{d}t}$  as the forecast horizon increases: it misses crucial details for the second component  $\frac{\mathrm{d}w}{\mathrm{d}t}$  which makes the forecast diverge from the ground truth. APHYNITY incorporates a Laplacian term as well as the data-driven  $F_{a}$  thus capturing the damping phenomenon and succeeding in maintaining physically sound results for long term forecasts, unlike Neural ODE.

Extension to non-stationary dynamics We provide additional results in Appendix G to tackle datasets where physical parameters of the equations vary in each sequence. To this end, we design an encoder able to perform parameter estimation for each sequence. Results show that APHYNITY accommodates well to this setting, with similar trends as those reported in this section.

# 5 DISCUSSION AND CONCLUSION

In this work, we introduce the APHYNITY framework that can efficiently augment approximate physical models with deep data-driven networks, performing similarly to models for which the underlying dynamics are entirely known. We exhibit the superiority of APHYNITY over data-driven, incomplete physics, and state-of-the-art approaches combining ML and MB methods, both in terms of forecasting and parameter identification on three various classes of physical systems. Besides, APHYNITY is flexible enough to adapt to different approximation levels of prior physical knowledge.

An appealing perspective is the applicability of APHYNITY on partially-observable settings. As a first step towards this goal, we evaluate APHYNITY on the Moving MNIST video prediction task (Srivastava et al., 2015). We show that we can reach new state-of-the-art performances on this dataset (see Appendix H Table 10), by using the APHYNITY augmentation framework on top on the recent PhyDNet architecture (Le Guen & Thome, 2020). It shows that the MB/ML cooperation brought up by APHYNITY can be successfully leveraged in a latent space for improving performances. We hope that the APHYNITY framework will open up the way to the design of a wide range of more flexible MB/ML models, e.g. in climate science, robotics or reinforcement learning. In particular, analyzing the theoretical decomposition properties in a partially-observed setting is an important direction for future work.

# REFERENCES

Ibrahim Ayed, Nicolas Cedilnik, Patrick Gallinari, and Maxime Sermesant. Ep-net: Learning cardiac electrophysiology models for physiology-based constraints in data-driven predictions. In Yves Coudiere, Valery Ozenne, Edward J. Vigmond, and Nejib Zemzemi (eds.), Functional Imaging and Modeling of the Heart - 10th International Conference, FIMH 2019, Bordeaux, France, June 6-8, 2019, Proceedings, volume 11504 of Lecture Notes in Computer Science, pp. 55-63. Springer, 2019a.  
Ibrahim Ayed, Emmanuel de Bézenac, Arthur Pajot, Julien Brajard, and Patrick Gallinari. Learning dynamical systems from partial observations. arXiv preprint arXiv:1902.11136, 2019b.  
Philipp Becker, Harit Pandya, Gregor Gebhardt, Cheng Zhao, James Taylor, and Gerhard Neumann. Recurrent kalman networks: Factorized inference in high-dimensional deep feature spaces. International Conference on Machine Learning (ICML), 2019.  
Dimitri P. Bertsekas. Constrained Optimization and Lagrange Multiplier Methods (Optimization and Neural Computation Series). Athena Scientific, 1 edition, 1996.  
Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932-3937, 2016.  
Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K. Duvenaud. Neural ordinary differential equations. In Advances in neural information processing systems (NeurIPS), pp. 6571-6583, 2018.  
Zhengdao Chen, Jianyu Zhang, Martin Arjovsky, and Léon Bottou. Symplectic recurrent neural networks. International Conference on Learning Representations (ICLR), 2020.  
Edward Choi, Mohammad Taha Bahadori, Jimeng Sun, Joshua Kulas, Andy Schuetz, and Walter Stewart. RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism. In Advances in Neural Information Processing Systems (NeurIPS), pp. 3504-3512, 2016.  
Philippe Courtier, J-N Thépaut, and Anthony Hollingsworth. A strategy for operational implementation of 4d-var, using an incremental approach. Quarterly Journal of the Royal Meteorological Society, 120(519):1367-1387, 1994.  
Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, and Shirley Ho. Lagrangian neural networks. *ICLR* 2020 Deep Differential Equations Workshop, 2020.  
Emmanuel de Bezenac, Arthur Pajot, and Patrick Gallinari. Deep learning for physical processes: Incorporating prior scientific knowledge. International Conference on Learning Representations (ICLR), 2018.  
John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. Journal of computational and applied mathematics, 6(1):19-26, 1980.  
James Fletcher and Warren Moors. Chebyshev sets. Journal of the Australian Mathematical Society, 98:161-231, 04 2014. doi: 10.1017/S1446788714000561.  
P. Gentine, M. Pritchard, S. Rasp, G. Reinaudi, and G. Yacalis. Could machine learning break the convection parameterization deadlock? Geophysical Research Letters, 45(11):5742-5751, 2018.  
Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian neural networks. In Advances in Neural Information Processing Systems (NeurIPS), pp. 15353-15363, 2019.  
Jun-Ting Hsieh, Bingbin Liu, De-An Huang, Li F Fei-Fei, and Juan Carlos Niebles. Learning to decompose and disentangle representations for video prediction. In Advances in Neural Information Processing Systems (NeurIPS), pp. 517-526, 2018.  
Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In Advances in Neural Information Processing Systems (NeurIPS), pp. 12519-12530, 2019.

Gordon G Johnson. A nonconvex set which has the unique nearest point property. Journal of Approximation Theory, 51(4):289 - 332, 1987.  
Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. 1960.  
Gene A. Klaasen and William C. Troy. Stationary wave solutions of a system of reaction-diffusion equations derived from the fitzhugh-nagumo equations. SIAM Journal on Applied Mathematics, 44(1):96-110, 1984. doi: 10.1137/0144008.  
William Large and Stephen Yeager. Diurnal to decadal global forcing for ocean and sea-ice models: The data sets and flux climatologies, 05 2004.  
Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In Computer Vision and Pattern Recognition (CVPR). 2020.  
Yun Long, Xueyuan She, and Saibal Mukhopadhyay. Hybridnet: integrating model-based and data-driven learning to predict evolution of dynamical systems. Conference on Robot Learning (CoRL), 2018a.  
Zichao Long, Yiping Lu, Xianzhong Ma, and Bin Dong. PDE-Net: Learning PDEs from data. In International Conference on Machine Learning (ICML), 2018b.  
Viraj Mehta, Ian Char, Willie Neiswanger, Youngseog Chung, and Jeff Schneider. Neural dynamical systems. ICLR 2020 Deep Differential Equations Workshop, 2020.  
Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. International Conference on Learning Representations (ICLR), 2020.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems 32, pp. 8024-8035. 2019.  
Pascal Pernot and Fabien Cailliez. A critical review of statistical calibration/prediction models handling data inconsistency and model inadequacy. *AChE Journal*, 63(10):4642-4665, 2017.  
Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 473:686-707, 2019.  
Markus Reichstein, Gustau Camps-Valls, Bjorn Stevens, Martin Jung, Joachim Denzler, Nuno Carvalhais, and & Prabhat. Deep learning and process understanding for data-driven Earth system science. Nature, 566:195-204, 2019.  
David Rolnick, Priya L Donti, Lynn H Kaack, Kelly Kochanski, Alexandre Lacoste, Kris Sankaran, Andrew Slavin Ross, Nikola Milojevic-Dupont, Natasha Jaques, Anna Waldman-Brown, et al. Tackling climate change with machine learning. In NeurIPS 2019 workshop on Climate Change with Machine Learning, 2019.  
Priyabrata Saha, Saurabh Dash, and Saibal Mukhopadhyay. PHICNet: Physics-incorporated convolutional recurrent neural networks for modeling dynamical systems. arXiv preprint arXiv:2004.06243, 2020.  
Sungyong Seo, Chuizheng Meng, and Yan Liu. Physics-aware difference graph networks for sparsely-observed dynamics. International Conference on Learning Representations (ICLR), 2020.  
Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (NeurIPS), pp. 802-810, 2015.

Justin Sirignano and Konstantinos Spiliopoulos. Dgm: A deep learning algorithm for solving partial differential equations. Journal of computational physics, 375:1339-1364, 2018.  
Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. Unsupervised learning of video representations using LSTMs. In International Conference on Machine Learning (ICML), pp. 843-852, 2015.  
Peter Toth, Danilo Jimenez Rezende, Andrew Jaegle, Sébastien Racanière, Aleksandar Botev, and Irina Higgins. Hamiltonian generative networks. International Conference on Learning Representations (ICLR), 2020.  
Jean-François Toubeau, Jérémie Bottieau, François Vallée, and Zacharie De Grève. Deep learning-based multivariate probabilistic forecasting for short-term scheduling in power markets. IEEE Transactions on Power Systems, 34(2):1203-1215, 2018.  
Benjamin Ummenhofer, Lukas Prantl, Nils Thuerey, and Vladlen Koltun. Lagrangian fluid simulation with continuous convolutions. International Conference on Learning Representations (ICLR), 2020.  
Qi Wang, Feng Li, Yi Tang, and Yan Xu. Integrating model-driven and data-driven methods for power system frequency stability assessment and control. IEEE Transactions on Power Systems, 34(6):4557-4568, 2019a.  
Yunbo Wang, Mingsheng Long, Jianmin Wang, Zhifeng Gao, and Philip S Yu. PredRNN: Recurrent neural networks for predictive learning using spatiotemporal LSTMs. In Advances in Neural Information Processing Systems (NeurIPS), pp. 879-888. 2017.  
Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang, and Philip S. Yu. PredRNN++: Towards a resolution of the deep-in-time dilemma in spatiotemporal predictive learning. In International Conference on Machine Learning (ICML), 2018.  
Yunbo Wang, Lu Jiang, Ming-Hsuan Yang, Li-Jia Li, Mingsheng Long, and Li Fei-Fei. Eidetic 3D LSTM: A model for video prediction and beyond. In International Conference on Learning Representations (ICLR), 2019b.  
Yunbo Wang, Jianjin Zhang, Hongyu Zhu, Mingsheng Long, Jianmin Wang, and Philip S Yu. Memory in memory: A predictive neural network for learning higher-order non-stationarity from spatiotemporal dynamics. In Computer Vision and Pattern Recognition (CVPR), pp. 9154-9162, 2019c.
