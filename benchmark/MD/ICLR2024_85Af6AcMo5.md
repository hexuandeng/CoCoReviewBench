# SCIRE-SOLVER: ACCELERATING DIFFUSION MODELS SAMPLING BY SCORE-INTEGRAND SOLVER WITH RECURSIVE DIFFERENCE

Anonymous authors

Paper under double-blind review

# ABSTRACT

One downside of Diffusion models (DMs) is their slow iterative process. Recent algorithms for fast sampling are designed from the differential equations. However, in the fast algorithms, estimating the derivative of the score function evaluations becomes intractable due to the complexity of large-scale, well-trained neural networks. In this work, we introduce the recursive difference method to calculate the derivative of the score function networks. Building upon, we propose SciRE-Solver with the convergence order guarantee for accelerating DMs sampling. Our proposed sampling algorithms attain SOTA FIDs in comparison to existing training-free sampling algorithms, under various numbers of score function evaluations (NFE). Such as, we achieve 3.48 FID with 12 NFE, and 2.42 FID with 20 NFE for continuous-time model on CIFAR-10. Moreover, we also test the pretrained model of EDM on CIFAR-10 and achieve 2.29 FID with 12 NFE, as well as 1.76 FID with 100 NFE. Empirically, SciRE-Solver with multi-step methods can achieve high-quality samples on the text-to-image generation tasks with only 6~20 NFEs.

# 1 INTRODUCTION

Diffusion models (DMs) (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021c) have recently gained significant progress on various tasks, including image generation (Dhariwal & Nichol, 2021; Meng et al., 2022), text-to-image generation (Ramesh et al., 2022), video synthesis (Ho et al., 2022), and voice synthesis (Chen et al., 2021; Liu et al., 2022a). DMs are composed of two diffusion stages. The forward stage of DMs is to add randomness with Gaussian noise in order to slowly disrupt the data distribution, without any training. The reverse stage of DMs is tasked with recovering the original input data from the diffused data by learning to reverse the forward diffusion process, step by step. DMs learn models by emulating the ground-truth inverse process of a fixed forward process.

One key downside of DMs is their slow iterative sampling process (Song et al., 2021a; Karras et al., 2022). Two distinct categories of methods have arisen to tackle this challenge: training-based and training-free methods. Training-based methods require additional training, such as knowledge distillation (Salimans & Ho, 2021; Meng et al., 2023) and consistency models (Song et al., 2023), noise level learning (Nichol & Dhariwal, 2021), or models combined with other generative models (Xiao et al., 2022; Vahdat et al., 2021a; Zhang & Chen, 2021). Training-free methods strive to accelerate the sampling process through numerical algorithms without requiring extra training. Recent training-free fast sampling methods can be attributed to the design of numerical algorithms for solving diffusion ODEs, benefiting from the fact that the sampling process of DMs can be reformulated as solving the corresponding diffusion ODE, as confirmed by DDIM (Song et al., 2021a) and Score-based models (Song et al., 2021c). Following this framework, several fast numerical algorithms with impressive results on DMs have been suggested, including PNDM (Liu et al., 2022b), DPM-Solver (Lu et al., 2022b), DEIS (Zhang & Chen, 2023), UniPC (Zhao et al., 2023), and ERA-Solver (Li et al., 2023). The core differences of these algorithms can be attributed to various derivative estimation or discretization methods, which imply that employing different methods to estimate the derivative of the score function will result in varying sampling performance.

In this work, we introduce a new derivative estimation method, called the Recursive Difference (RD), to calculate the derivative of the score function networks. The FID-measured ablation experiments demonstrate the effectiveness of using the RD method. Based on the RD method and the truncated Taylor expansion of score-integrand, we propose SciRE-Solver with the convergence order guarantee

![](images/b9526c5225347c41bc48444a3efabbb0e82b4c6b49817612005e2ed93e299440.jpg)  
Figure 1: Generated samples of the pre-trained DM on ImageNet  $256 \times 256$  (classifier scale: 2.5) using 10-50 sampling steps from different sampling methods with the same random seed and codebase. Our algorithm, SciRE-V1 Solver, generates high-quality results in a fewer number of steps.

for accelerating DMs sampling. Our proposed sampling algorithms with RD method advance the sampling efficiency of the training-free sampling method to a new level. Such as, we achieve 3.48 FID with 12 NFE and 2.42 FID with 20 NFE for continuous-time DMs on CIFAR-10, respectively. Furthermore, we observe that SciRE-V1 with a small NFEs demonstrates the promising potential to surpass the FIDs achieved in the original papers of some pre-trained models, distinguishing itself from other samplers. For example, we reach SOTA value of 2.40 FID with 100 NFE for continuous-time DM and of 3.15 FID with 84 NFE for discrete-time DM on CIFAR-10, as well as of 2.17 (2.02) FID with 18 (50) NFE for discrete-time DM on CelebA  $64 \times 64$ . Experiments demonstrate that SciRE-Solver (V1 and V2) exhibit also the ability to generate high-quality results with fewer iterations when applied to high-resolution image datasets, as shown in Figures 1, 6, 8.

# 2 BACKGROUND

# 2.1 DIFFUSION ODEs

A Markov sequence  $\{\mathbf{x}_t\}_{t\in [0,T]}$  with  $T > 0$  starting with  $\mathbf{x}_0$ , in the forward diffusion of DMs for  $D$ -dimensional data, is defined by the following transition kernel:

$$
q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right) = \mathcal {N} \left(\mathbf {x} _ {t}; \alpha_ {t} \mathbf {x} _ {0}, \sigma_ {t} ^ {2} \mathbf {I}\right). \tag {2.1}
$$

This transition kernel is equivalent to the stochastic differential equation (SDE) (Kingma et al., 2021):

$$
\mathrm {d} \mathbf {x} _ {t} = f (t) \mathbf {x} _ {t} \mathrm {d} t + g (t) \mathrm {d} \omega_ {t}, \quad \mathbf {x} _ {0} \sim q _ {0} \left(\mathbf {x} _ {0}\right), \tag {2.2}
$$

where  $\omega_{t} \in \mathbb{R}^{D}$  denotes a standard Wiener process, and  $f(t) = \frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}$ ,  $g^2 (t) = \frac{\mathrm{d}\sigma_t^2}{\mathrm{d}t} - 2\frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}\sigma_t^2$ . This forward diffusion has the following equivalent reverse diffusion from time  $T$  to 0 (Song et al., 2021c):

$$
\mathrm {d} \mathbf {x} _ {t} = \left[ f (t) \mathbf {x} _ {t} - g ^ {2} (t) \nabla_ {\mathbf {x}} \log q _ {t} (\mathbf {x} _ {t}) \right] \mathrm {d} t + g (t) \mathrm {d} \overline {{\omega}} _ {t}, \quad \mathbf {x} _ {T} \sim q _ {T} (\mathbf {x} _ {T}), \tag {2.3}
$$

where  $\overline{\omega}_t$  represents a standard Wiener process. In score-based models, Song et al. (2021c) derived the following ordinary differential equation (ODE):

$$
\frac {\mathrm {d} \mathbf {x} _ {t}}{\mathrm {d} t} = f (t) \mathbf {x} _ {t} - \frac {1}{2} g ^ {2} (t) \nabla_ {\mathbf {x}} \log q _ {t} \left(\mathbf {x} _ {t}\right), \quad \mathbf {x} _ {T} \sim q _ {T} \left(\mathbf {x} _ {T}\right), \tag {2.4}
$$

where the marginal distribution  $q_{t}(\mathbf{x}_{t})$  of  $\mathbf{x}_t$  is equivalent to the marginal distribution of  $\mathbf{x}_t$  of the SDE in Eq. (2.3). By substituting the trained noise prediction model  $\epsilon_{\theta}(\mathbf{x}_t,t)$  for the scaled score function:  $-\sigma_t\nabla_{\mathbf{x}}\log q_t(\mathbf{x}_t)$ , Song et al. (2021c) defined the diffusion ODE for DMs:

$$
\frac {\mathrm {d} \mathbf {x} _ {t}}{\mathrm {d} t} = f (t) \mathbf {x} _ {t} + \frac {g ^ {2} (t)}{2 \sigma_ {t}} \epsilon_ {\theta} (\mathbf {x} _ {t}, t), \quad \mathbf {x} _ {T} \sim \mathcal {N} (\mathbf {0}, \hat {\sigma} ^ {2} \boldsymbol {I}). \tag {2.5}
$$

Since the data prediction model  $\pmb{x}_{\theta}(\mathbf{x}_t,t)$  and the noise prediction model  $\epsilon_{\theta}(\mathbf{x}_t,t)$  satisfying:  $\pmb{x}_{\theta}(\mathbf{x}_t,t) = (\mathbf{x}_t - \sigma_t\pmb{\epsilon}_{\theta}(\mathbf{x}_t,t)) / \alpha_t$  (Kingma et al., 2021), there exists an equivalent diffusion ODE:

$$
\frac {\mathrm {d} \mathbf {x} _ {t}}{\mathrm {d} t} = \left(f (t) + \frac {g ^ {2} (t)}{2 \sigma_ {t} ^ {2}}\right) \mathbf {x} _ {t} - \alpha_ {t} \frac {g ^ {2} (t)}{2 \sigma_ {t} ^ {2}} \boldsymbol {x} _ {\theta} (\mathbf {x} _ {t}, t), \quad \mathbf {x} _ {T} \sim \mathcal {N} (\mathbf {0}, \hat {\sigma} ^ {2} \boldsymbol {I}). \tag {2.6}
$$

# 2.2 NUMERICAL METHODS OF DIFFUSION ODEs

Traditional numerical techniques for solving ODEs find their roots in concepts like Taylor expansions, the trapezoidal rule, and Simpson's rule. These foundational ideas have paved the way for the development of well-known approaches such as Euler's method, Runge-Kutta methods, and linear multi-step methods (Suli, 2010). In the realm of diffusion ODEs, a similar lineage of inspiration from these classical methods can be observed in the construction of various numerical approaches.

DDIM (Song et al., 2021a) can be accurately interpreted as the forward Euler method from the perspective of the diffusion ODE in Eq. 2.5. Song et al. (2021c) tested the Runge-Kutta Fehlberg method for diffusion ODEs. Liu et al. (2022b) investigated the Runge-Kutta methods and linear multi-step methods, and based on this, further proposed the PNDM. Lu et al. (2022b) introduced the exponential integrator with the semi-linear structure from the ODE literature (Atkinson et al., 2011), and employed Taylor expansion techniques to handle the remaining integration, resulting in the proposed DPM-Solver. Zhang & Chen (2023) proposed DEIS by introducing the exponential integrator and further leveraging the assistance of both Runge-Kutta methods and linear multi-step (Adams-Bashforth) methods. Li et al. (2023) explored the use of linear multi-step (implicit Adams) methods with Lagrange interpolation function, and further proposed ERA-Solver.

In this work, our main focus is on algorithms based on Taylor expansions. We introduce sampling algorithms that are predicated on the recursive difference method, which stands out as one of the distinctions between our algorithm and the DPM-Solver.

# 3 SAMPLING ALGORITHMS BASED ON RECURSIVE DIFFERENCE FOR DIFFUSION MODELS

This section introduces the recursive difference (RD) method, which is employed to compute the derivative of score function within sampling algorithms for DMs based on Taylor expansion. Based on the RD method and the truncated Taylor expansion of the score-integrand, we propose the SciRE-Solver with the convergence order guarantee to accelerating sampling of DMs.

# 3.1 RECURSIVE DIFFERENCE METHOD FORDIFFUSION ODEs

Since samples can be generated by solving the diffusion ODEs numerically from  $T$  to 0, sampling algorithms can be designed from the numerical solutions of differential equations. By applying the variation-of-constants formula (Hale & Lunel, 2013) to ODEs (2.5) and (2.6), we have

$$
\mathbf {x} _ {t} = e ^ {\int_ {s} ^ {t} f (r) \mathrm {d} r} \left(\int_ {s} ^ {t} h _ {1} (r) \epsilon_ {\theta} (\mathbf {x} _ {r}, r) \mathrm {d} r + \mathbf {x} _ {s}\right), \quad \mathbf {x} _ {t} = e ^ {h _ {2} (t)} \left(- \int_ {s} ^ {t} e ^ {- h _ {2} (r)} \frac {\alpha_ {r} g ^ {2} (r)}{2 \sigma_ {r} ^ {2}} \mathbf {x} _ {\theta} (\mathbf {x} _ {r}, r) \mathrm {d} r + \mathbf {x} _ {s}\right), \tag {3.1}
$$

where  $h_1(r) \coloneqq e^{-\int_s^r f(z)\mathrm{d}z}\frac{g^2(r)}{2\sigma_r}$ ,  $h_2(r) \coloneqq \int_s^r f(z) + \frac{g^2(z)}{2\sigma_z^2}\mathrm{d}z$ , and  $\mathbf{x}_s$  represents the given initial value. Then, the most simplified solution formulas for the diffusion ODEs can be obtained, as follows.

Proposition 3.1 Let  $\mathbf{x}_s$  be a given initial value at time  $s > 0$ . Then, the diffusion ODEs in Eq. (2.5) and Eq. (2.6) has the following solution formulas, respectively:

$$
\frac {\mathbf {X} _ {t}}{\alpha_ {t}} - \frac {\mathbf {X} _ {s}}{\alpha_ {s}} = \int_ {\mathrm {N S R} (s)} ^ {\mathrm {N S R} (t)} \epsilon_ {\theta} \left(\mathbf {x} _ {\mathrm {r N S R} (\tau)}, \mathrm {r N S R} (\tau)\right) \mathrm {d} \tau , \tag {3.2}
$$

$$
\frac {\mathbf {X} _ {t}}{\sigma_ {t}} - \frac {\mathbf {X} _ {s}}{\sigma_ {s}} = \int_ {1 / \mathrm {N S R} (s)} ^ {1 / \mathrm {N S R} (t)} x _ {\theta} \left(\mathbf {x} _ {\mathrm {r N S R} (1 / \tau)}, \mathrm {r N S R} (1 / \tau)\right) \mathrm {d} \tau , \tag {3.3}
$$

where  $\mathrm{NSR}(\gamma) \coloneqq \frac{\sigma_{\gamma}}{\alpha_{\gamma}}$ , we refer to it as the time-dependent noise-to-signal-ratio (NSR) function;  $\mathrm{rNSR}(\cdot)$  is the inverse function of  $\mathrm{NSR}(\cdot)$ , satisfying  $\gamma = \mathrm{rNSR}(\mathrm{NSR}(\gamma))$  for any diffusion time  $\gamma$ . We provide the detailed derivation in Appendix B for two solution formulas.

As the integral term in the r.h.s. of (3.2) is solely dependent on the evaluation network  $\epsilon_{\theta}(\mathbf{x}_s,s)$  of scaled score function, we refer to such a concise solution formula as "score-integrand form" of diffusion ODEs. Compared to the exponential-product-score-based solution formula in (Lu et al., 2022b), empirically, the algorithm based on the score-integrand form generates more stable samples when using a few NFEs ( $\leq 10$ ), as shown in Figures 1 and 5. In score-integrand form, we can solve

the diffusion ODE by directly integrating the  $\epsilon_{\theta}(\mathbf{x}_{\mathrm{rNSR}(\tau)},\mathrm{rNSR}(\tau))$ . In theory, directly tackling this problem is very challenging because  $\epsilon_{\theta}(\mathbf{x}_t,t)$  is a large-scale, well-trained complex neural network. Nevertheless, we can solve it using numerical methods. For example, we can perform a Taylor expansion on the score-integrand to obtain a rough iterative scheme.

Denote  $h_{t_i} \coloneqq \mathrm{NSR}(t_{i-1}) - \mathrm{NSR}(t_i)$ ,  $\tau_{t_i} \coloneqq \mathrm{NSR}(t_i)$ ,  $\psi(\tau) \coloneqq \mathrm{rNSR}(\tau)$ , and  $\epsilon_\theta^{(k)}\left(\mathbf{x}_{\psi(\tau)}, \psi(\tau)\right) \coloneqq \frac{\mathrm{d}^k \epsilon_\theta\left(\mathbf{x}_{\psi(\tau)}, \psi(\tau)\right)}{\mathrm{d} \tau^k}$  as  $k$ -th order total derivative of  $\epsilon_\theta\left(\mathbf{x}_{\psi(\tau)}, \psi(\tau)\right)$  w.r.t.  $\tau$ . For  $n \geq 1$ , the  $n$ -th order Taylor expansion of  $\epsilon_\theta\left(\mathbf{x}_{\psi(\tau_{t_{i-1}})}, \psi(\tau_{t_{i-1}})\right)$  w.r.t.  $\tau$  at  $\tau_{t_i}$  is

$$
\epsilon_ {\theta} \left(\mathbf {x} _ {\psi \left(\tau_ {t _ {i - 1}}\right)}, \psi \left(\tau_ {t _ {i - 1}}\right)\right) = \sum_ {k = 0} ^ {n} \frac {h _ {t _ {i}} ^ {k}}{k !} \epsilon_ {\theta} ^ {(k)} \left(\mathbf {x} _ {\psi \left(\tau_ {t _ {i}}\right)}, \psi \left(\tau_ {t _ {i}}\right)\right) + O \left(h _ {t _ {i}} ^ {n + 1}\right). \tag {3.4}
$$

By substituting this Taylor expansion into Eq. (3.2), we get

$$
\mathbf {x} _ {t _ {i - 1}} = \frac {\alpha_ {t _ {i - 1}}}{\alpha_ {t _ {i}}} \mathbf {x} _ {t _ {i}} + \alpha_ {t _ {i - 1}} \sum_ {k = 0} ^ {n} \frac {h _ {t _ {i}} ^ {k + 1}}{(k + 1) !} \boldsymbol {\epsilon} _ {\theta} ^ {(k)} \left(\mathbf {x} _ {\psi \left(\tau_ {t _ {i}}\right)}, \psi \left(\tau_ {t _ {i}}\right)\right) + O \left(h _ {t _ {i}} ^ {n + 2}\right). \tag {3.5}
$$

Consequently, Eq. (3.5) provides an iterative scheme for solving the diffusion ODE. By following the classical thought path, we can develop an  $n$ -th order solver for diffusion ODEs by omitting the error term  $\mathcal{O}(h_{t_i}^{n + 1})$  and approximating the first  $(n - 1)$ -order derivatives  $\epsilon_{\theta}^{(k)}(\mathbf{x}_{\psi (\tau_{t_i})},\psi (\tau_{t_i}))$  for  $k\leq n - 1$  in turn (Atkinson et al., 2011). Such as, we can obtain the first-order iterative algorithm when  $n = 1$ :

$$
\tilde {\mathbf {x}} _ {t _ {i - 1}} = \frac {\alpha_ {t _ {i - 1}}}{\alpha_ {t _ {i}}} \tilde {\mathbf {x}} _ {t _ {i}} + \alpha_ {t _ {i - 1}} h _ {i} \epsilon_ {\theta} \left(\tilde {\mathbf {x}} _ {\psi \left(\tau_ {t _ {i}}\right)}, \psi \left(\tau_ {t _ {i}}\right)\right), \tag {3.6}
$$

where  $\tilde{\mathbf{x}}$  is an approximation of the true value  $\mathbf{x}$ , and  $\tilde{\mathbf{x}}_{t_N} = \mathbf{x}_T$  is the given initial value.

Beneath the surface of smooth operations, a pivotal challenge emerges: how to assess derivatives in Taylor expansions when dealing with  $n \geq 2$ . When it comes to estimating derivatives, one preferred choice is the finite difference (FD) method. Clearly, the FD method truncates all challenging higher-order derivative terms ( $k \geq 2$ ) and possesses a truncation error of  $O(h_{i_i})$ . Some indications suggest that the FD method often lacks outstanding numerical performance in practice. For example, in the pursuit of enhanced numerical performance, it is common to replace  $(e^{h} - h - 1) / h^{2}$  with  $(e^{h} - 1) / h$  as the new FD coefficient within the framework of exponential integrators (Hochbruck & Ostermann, 2005; Lu et al., 2022b; Zhang & Chen, 2023), guided by the concept of equivalent infinitesimal w.r.t  $h$ . In light of such indication, we speculate that utilizing the conventional FD method directly to evaluate the derivative of the score function may be a suboptimal choice. Our experiments have further substantiated this conjecture, as illustrated in Figure 3.

To improve the FD method while avoiding the intricacies of higher-order derivatives, we recursively apply the principles of FD to handle terms involving higher-order derivatives at the evaluation point. For example, when dealing with third-order derivative terms, our approach is outlined as follows:

$$
\begin{array}{l} \Gamma^ {(3)} \left(\tau_ {t _ {i}}\right) = \frac {\Gamma^ {(2)} \left(\tau_ {t _ {i - 1}}\right) - \Gamma^ {(2)} \left(\tau_ {t _ {i}}\right)}{h _ {t _ {i}}} + O \left(h _ {t _ {i}}\right) = \frac {\Gamma^ {(2)} \left(\tau_ {t _ {i - 1}}\right)}{h _ {t _ {i}}} - \frac {\frac {\Gamma^ {(1)} \left(\tau_ {t _ {i - 1}}\right) - \Gamma^ {(1)} \left(\tau_ {t _ {i}}\right)}{h _ {t _ {i}}}}{h _ {t _ {i}}} + O \left(h _ {t _ {i}}\right) \tag {3.7} \\ = \frac {\Gamma^ {(1)} (\tau_ {t _ {i}})}{h _ {t _ {i}} ^ {2}} - \frac {\Gamma^ {(1)} (\tau_ {t _ {i - 1}})}{h _ {t _ {i}} ^ {2}} + \frac {\Gamma^ {(2)} (\tau_ {t _ {i - 1}})}{h _ {t _ {i}}} + \mathcal {O} (h _ {t _ {i}}), \\ \end{array}
$$

where  $\tau_{t_i}$  represents the evaluation point and  $\Gamma^{(k)}(\tau)$  is used to denote  $\epsilon_{\theta}^{(k)}\left(\mathbf{x}_{\psi (\tau)},\psi (\tau)\right)$  for simplicity. Under such recursive rule, each high-order derivative term in Eq. (3.4) can be rewritten as the sum of a scaled first-order derivative function at  $\tau_{t_i}$  and a function w.r.t.  $\tau_{t_{i - 1}}$  , while this representation incurs a truncation error of  $O(h_{t_i})$  . Subsequently, by merging the resulting series of scaled first-order derivatives, we can obtain a new derivative estimate for the score function. We refer to such structured estimation method as the recursive difference (RD) method. In Appendix D, we present a detailed derivation of the RD method, with the results as stated in Theorem 3.1 and Corollary 3.1.

Denote  $\mathrm{NSR}_{\min} := \min_{i} \{\mathrm{NSR}(t_i)\}$ ,  $\mathrm{NSR}_{\max} := \max_{i} \{\mathrm{NSR}(t_i)\}$ . We derive the following recursive results for the derivative at the evaluation point.

![](images/7aed8ac15bc9be545d7c58887e6a456df0e8fd2f43345bb88187a3017fbc0355.jpg)  
Figure 2: Schematic diagram of the recursive difference method tailored for sampling algorithms of diffusion models. The diagram exhibits the derivative process of  $\Gamma^{(1)}(\tau_s)$  with  $\Gamma^{(0)}(\tau_t)$  given as input. Similarly, we can obtain the  $\Gamma^{(k)}(\tau_s), \forall k \in \mathbb{Z}_+$  with  $\Gamma^{(0)}(\tau_t)$  as input using analogous procedures.

Theorem 3.1 Let  $\mathbf{x}_s$  be a given initial value at time  $s > 0$ ,  $\mathbf{x}_t$  be the estimated value at time  $t$  obtained by the first-order iterative algorithm in Eq. (3.6). Assume that  $\epsilon_{\theta}\left(\mathbf{x}_{\psi (\tau)},\psi (\tau)\right)\in \mathbb{C}^{\infty}[\mathrm{NSR}_{\min},\mathrm{NSR}_{\max}]$ . Then, we have

$$
\begin{array}{l} \epsilon_ {\theta} ^ {(1)} \left(\tilde {\mathbf {x}} _ {\psi (\tau_ {s})}, \psi (\tau_ {s})\right) = \frac {e}{e - 1} \frac {\epsilon_ {\theta} \left(\tilde {\mathbf {x}} _ {\psi (\tau_ {t})} , \psi (\tau_ {t})\right) - \epsilon_ {\theta} \left(\tilde {\mathbf {x}} _ {\psi (\tau_ {s})} , \psi (\tau_ {s})\right)}{h _ {s}} \tag {3.8} \\ - \frac {\epsilon_ {\theta} ^ {(1)} (\tilde {\mathbf {x}} _ {\psi (\tau_ {t})} , \psi (\tau_ {t}))}{e - 1} - \frac {(e - 2) h _ {s}}{2 (e - 1)} \epsilon_ {\theta} ^ {(2)} (\tilde {\mathbf {x}} _ {\psi (\tau_ {t})}, \psi (\tau_ {t})) + O \left(h _ {s} ^ {2}\right), \\ \end{array}
$$

where  $\mathbb{C}^{\infty}[\mathrm{NSR}_{\mathrm{min}},\mathrm{NSR}_{\mathrm{max}}]$  denotes  $\epsilon_{\theta}\left(\mathbf{x}_{\psi (\tau)},\psi (\tau)\right)$  is an infinitely continuously differentiable function w.r.t.  $\tau$  over the interval  $[\mathrm{NSR}_{\mathrm{min}},\mathrm{NSR}_{\mathrm{max}}]$ .

We observe that the differentiability constraint imposed by Theorem 3.1 appears to be rather restrictive. To enhance its broad applicability, we further derive the recursive result with limited differentiability.

Corollary 3.1 Let  $\mathbf{x}_s$  be a given initial value at time  $s > 0$ ,  $\mathbf{x}_t$  be the estimated value at time  $t$  obtained by the first-order iterative algorithm in Eq. (3.6). Assume that  $\epsilon_{\theta}\left(\mathbf{x}_{\psi(\tau)}, \psi(\tau)\right) \in \mathbb{C}^m[\mathrm{NSR}_{\min}, \mathrm{NSR}_{\max}]$ , i.e.,  $m$  times continuously differentiable, where  $m \geq 3$ . Then, we have

$$
\begin{array}{l} \epsilon_ {\theta} ^ {(1)} \left(\tilde {\mathbf {x}} _ {\psi (\tau_ {s})}, \psi (\tau_ {s})\right) = \frac {1}{\phi_ {1} (m)} \frac {\epsilon_ {\theta} \left(\tilde {\mathbf {x}} _ {\psi (\tau_ {t})} , \psi (\tau_ {t})\right) - \epsilon_ {\theta} \left(\tilde {\mathbf {x}} _ {\psi (\tau_ {s})} , \psi (\tau_ {s})\right)}{h _ {s}} \tag {3.9} \\ - \frac {\phi_ {2} (m)}{\phi_ {1} (m)} \boldsymbol {\epsilon} _ {\theta} ^ {(1)} \left(\tilde {\mathbf {X}} _ {\psi (\tau_ {t})}, \psi (\tau_ {t})\right) - \frac {\phi_ {3} (m) h _ {s}}{\phi_ {1} (m)} \boldsymbol {\epsilon} _ {\theta} ^ {(2)} \left(\tilde {\mathbf {X}} _ {\psi (\tau_ {t})}, \psi (\tau_ {t})\right) + \mathcal {O} \left(h _ {s} ^ {2}\right), \\ \end{array}
$$

where  $\phi_1(m) = \sum_{k=1}^{m} \frac{(-1)^{k-1}}{k!}$ ,  $\phi_2(m) = \sum_{k=2}^{m} \frac{(-1)^k}{k!}$ , and  $\phi_3(m) = \sum_{k=3}^{m} \frac{(-1)^{k+1}}{k!}$ .

A simplified truncation form for the RD method is given by  $\Gamma^{(1)}(\tau_{t_i}) = \frac{\Gamma(\tau_{t_{i-1}}) - \Gamma(\tau_{t_i})}{\phi_1(m)h_{t_i}}$ , obtained by substituting the RD estimation formula from Eq. (3.9) into Eq. (3.5). The complete recursive process of such simplified form is illustrated in Figure 2. Further details are provided in Appendix E.1. Since other truncated forms of the RD method necessitate derivative evaluation at point  $\tau_t$  beyond the evaluation point  $\tau_s$ , we leave this exploration for future research. The main characteristic of such simplified RD version is that this novel estimation incorporates low-order derivative information hidden in the higher-order derivative terms of the Taylor expansion. Compared to FD method, such RD method incorporates additional information  $\frac{1 - \phi_1(m)}{\phi_1(m)}\frac{\Gamma(\tau_{t_i-1}) - \Gamma(\tau_{t_i})}{h_{t_i}}$  from other higher-order derivative terms, which may counterbalance with these higher-order terms to a certain level. In Appendix E.2, we provide essential analyses concerning the simplified RD method.

# 3.2 SAMPLING ALGORITHMS BASED ON RECURSIVE DIFFERENCE METHOD

Since Proposition 3.1 involves distinct differential equations that result in different discretization results, we propose two solver versions called SciRE-V1 and SciRE-V2 for the diffusion ODE

corresponding to the noise prediction model and the data prediction model, respectively. Now, based on the Eq. (3.5) and the RD methods stated by Corollary 3.1 and Theorem 3.1, we propose two algorithms named SciRE-V1-2 and SciRE-V1-3 for  $n = 2$  and  $n = 3$ , respectively. Under mild assumptions, we provide the convergence order for SciRE-V1- $k$  ( $k = 2, 3$ ), as stated in the following theorem. The proof is given in Appendix F. Due to the typically increased complexity of higher-order algorithms, the treatment of  $k \geq 4$  will be left for future research. The iteration schemes of SciRE-V1 using multi-step methods are provided in Appendix G.1. In Appendix G.2, we propose SciRE-V2 for the ODE Eq. (3.3) using the mentioned-above thought process and the RD method.

# Algorithm 1 SciRE-V1-2

Require: initial value  $\mathbf{x}_T$ , time trajectory  $\{t_i\}_{i=0}^N$ , model  $\epsilon_\theta, m \geq 3$

1:  $\tilde{\mathbf{x}}_{t_N}\gets \mathbf{x}_T,r_1\gets \frac{1}{2}$  
2: for  $i \gets N$  to 1 do

3:  $h_i \gets \mathrm{NSR}(t_{i-1}) - \mathrm{NSR}(t_i)$  
4:  $s_i \gets \mathrm{rNSR}(\mathrm{NSR}(t_i) + r_1h_i)$  
5:  $\tilde{\mathbf{x}}_{S_i} \leftarrow \frac{\alpha_{S_i}}{\alpha_{t_i}} \tilde{\mathbf{x}}_{t_i} + \alpha_{S_i} r_1 h_i \epsilon_\theta (\tilde{\mathbf{x}}_{t_i}, t_i)$  
6:  $\tilde{\mathbf{x}}_{t_{i - 1}}\gets \frac{\alpha_{t_{i - 1}}}{\alpha_{t_i}}\tilde{\mathbf{x}}_t_i + \alpha_{t_{i - 1}}h_i\epsilon_\theta (\tilde{\mathbf{x}}_{t_i},t_i) + \alpha_{t_{i - 1}}\frac{h_i}{2\phi_1(m)r_1} (\epsilon_\theta (\tilde{\mathbf{x}}_{s_i},s_i) - \epsilon_\theta (\tilde{\mathbf{x}}_{t_i},t_i))$  
7: end for

Return:  $\tilde{\mathbf{x}}_0$

# Algorithm 2 SciRE-V1-3

Require: initial value  $\mathbf{x}_T$ , time trajectory  $\{t_i\}_{i=0}^N$ , model  $\epsilon_{\theta}, m \geq 3$

1:  $\tilde{\mathbf{x}}_{t_N}\gets \mathbf{x}_T,r_1\gets \frac{1}{3},r_2\gets \frac{2}{3}$  
2: for  $i\gets N$  to 1 do  
3:  $h_i \gets \mathrm{NSR}(t_{i-1}) - \mathrm{NSR}(t_i)$  
4:  $s_{i_1}, s_{i_2} \gets \mathrm{rNSR}(\mathrm{NSR}(t_i) + r_1h_i), \mathrm{rNSR}(\mathrm{NSR}(t_i) + r_2h_i)$  
5:  $\tilde{\mathbf{X}}_{s_{i_1}}\gets \frac{\alpha_{s_{i_1}}}{\alpha_{t_i}}\tilde{\mathbf{X}}_{t_i} + \alpha_{s_{i_1}}r_1h_i\epsilon_\theta (\tilde{\mathbf{X}}_{t_i},t_i)$  
6:  $\tilde{\mathbf{x}}_{s_{i_2}}\gets \frac{\alpha_{s_{i_2}}}{\alpha_{t_i}}\tilde{\mathbf{x}}_{t_i} + \alpha_{s_{i_2}}r_2h_i\pmb {\epsilon}_\theta (\tilde{\mathbf{x}}_{t_i},t_i) + \alpha_{s_{i_2}}\frac{h_i}{\phi_1(m)}\left(\pmb {\epsilon}_\theta (\tilde{\mathbf{x}}_{s_{i_1}},s_{i_1}) - \pmb {\epsilon}_\theta (\tilde{\mathbf{x}}_{t_i},t_i)\right)$  
7:  $\tilde{\mathbf{x}}_{t_{i - 1}}\gets \frac{\alpha_{t_{i - 1}}}{\alpha_{t_i}}\tilde{\mathbf{x}}_{t_i} + \alpha_{t_{i - 1}}h_i\epsilon_\theta (\tilde{\mathbf{x}}_{t_i},t_i) + \alpha_{t_{i - 1}}\frac{h_i}{2\phi_1(m)r_2}\left(\epsilon_\theta (\tilde{\mathbf{x}}_{s_{i_2}},s_{i_2}) - \epsilon_\theta (\tilde{\mathbf{x}}_{t_i},t_i)\right)$  
8: end for

Return:  $\tilde{\mathbf{x}}_0$

Theorem 3.2 Assume that  $\epsilon_{\theta}\bigl (\mathbf{x}_{\psi (\tau)},\psi (\tau)\bigr)\in \mathbb{C}^{m}[\mathrm{NSR}_{\min},\mathrm{NSR}_{\max}]$ . Then, for  $k = 2,3$ , the global convergence order of SciRE-VI- $k$  is no less than  $k - 1$ .

# 4 ASSESSING THE EFFICACY OF THE RD METHOD THROUGH ABLATION STUDIES

This section demonstrates the effectiveness of the RD method from two perspectives: 1. Comparing it with traditional finite difference (FD) method; 2. Introducing the RD method into the exponential-based calculation formula and comparing it with its counterpart algorithm, DPM-Solver-2.

In Corollary 3.1, the RD method degenerates into the FD method, if we set  $\phi_1(m) = 1$  and drop other terms. Thus, we set  $\phi_1(m) = 1$  in our SciRE-V1 codebase to represent the sampling algorithm based on FD method. Comparative experiments are presented in Figure 3 under identical settings.

To further investigate the RD method, we introduce SciREI-Solver  $(n = 2)$ , a variant combining the RD method and the exponential-based calculation formula from DPM-Solver. Refer to Appendix C for the details of SciREI-Solver. We compare the generative performance of SciREI-Solver-2 and DPM-Solver-2 with the identical settings on the CIFAR-10 and CelebA 64 datasets using various time trajectories and termination times, the experiment results are presented in Figure 4. More generally, we also provide the sampling comparison between the RD-based sampling algorithms (SciRE-V1-2 and SciREI-Solver-2) and the baseline algorithm (DPM-Solver-2) on high-resolution image datasets, as shown in Figure 5. More comparisons are provided in Appendix C.

![](images/4b856084f41c6f22e87929286c4cfbc0f22baf8ba8df7f97155b843f46e83b5a.jpg)  
(a) CIFAR-10 (discrete)

![](images/b85c223cc3547fb82a1ce4b53619c2296ab11d5d83943dbda45412520af10c5d.jpg)  
(b) CelebA  $64\times 64$  (discrete)

![](images/db6cf3f8a8d788f70e03b4405bbf26693bf03dc639da78d3ecd507263caf747e.jpg)  
Figure 3: Comparisons of FID  $\downarrow$  obtained by employing RD and FD in SciRE-V1 codebase. The RD-based method is consistently superior to the FD-based method across different cases.  
(a) CIFAR-10  $(1e - 3)$

![](images/be897c76737ec5ebb9a19a6aaa420459e68043b04d4688b9fb3a45d81a2b0e28.jpg)  
(b) CIFAR-10  $(1e - 4)$  
Figure 4: Comparisons of FID  $\downarrow$  obtained by SciREI-2 and DPM-2 solvers across different trajectories. SciREI-2 is more robust than DPM-2 across different time trajectories under the same sampling step.

![](images/2b6d14d4a26f890a2c1055d730609415d316b034ce1a8ef7f8ecb933315018d6.jpg)  
(c) CelebA 64 (1e-3)

![](images/89c30cac5e0bc53162d693f6d9b6773317185322a0e78740327b1c783f675c59.jpg)  
(d) CelebA 64 (1e-4)

# 5 EXPERIMENTS

This section shows that SciRE-Solver can improve the sampling efficiency of pre-trained DPM models, including continuous-time and discrete-time DMs. We conduct sampling experiments using individual SciRE-V1-2 and SciRE-V1-3 on the pre-trained models of DM. When comparing with existing fast sampling algorithms, we will compare the best FID values reported by these algorithms in the relevant literature with the FID obtained by our proposed SciRE-V1 under the same NFE, as shown in Table 1. Moreover, we also investigate the SciRE-Solver (V1 and V2) algorithms on image-text generation tasks, as shown in Figures 6 and 8. More details and experiments can be found in Appendix H.

# 5.1 EXPERIMENT SETTING AND ABLATION STUDY

When running our proposed SciRE-V1- $k$  in Algorithms 1 and 2, it is necessary to assign a value  $m$  to  $\phi_1(m)$ . As stated in Corollary 3.1, when assigning  $m$ , we need to ensure that  $m \geq 3$ . Considering that the limit of  $\phi_1(m)$  is  $\frac{e - 1}{e}$ , then our experiments only consider these two extreme cases, i.e., we only choose to allocate  $m$  as 3 or directly set  $\phi_1(m) = \frac{e - 1}{e}$ . We provide ablation experiments for these two cases in Appendix H. The earlier experiments were all run on TITAN-V GPUs.

# 5.2 COMPARISONS OF SAMPLING METHODS USING DISCRETE-TIME AND CONTINUOUS-TIME MODELS

We compare SciRE-V1 proposed in Section 3.2 with existing discrete-time training-free methods in Table 1. Specifically, we use the discrete-time model trained by  $L_{\text{simple}}$  in (Ho et al., 2020) on CIFAR-10 and CelebA  $64 \times 64$  datasets with linear noise schedule, and assign  $m = 3$  to  $\phi_1(m)$ . Under this setting, we use the same NSR-type time trajectory with fixed parameter for both SciRE-V1-2 and SciRE-V1-3, the details are available in Appendix H. SciRE-V1 almost reaches convergence at around 66 NFE and 18 NFE, achieving the new SOTA values of 3.15 FID with 84 NFE, and of 2.17 FID with 18 NFE on CIFAR-10 and CelebA  $64 \times 64$ , respectively.

We compare SciRE-V1- $k$  and SciRE-V1-agile with DPM-Solver- $k$  (Lu et al., 2022b), DPM-Solver-fast and DEIS (Zhang & Chen, 2023), where  $k = 2$ , 3. On CIFAR-10, we use "VP deep" model (Song et al., 2021c) with the linear noise schedule. When NFE  $\geq 15$ , we employ the identical NSR-type time trajectories with consistent parametric functions for SciRE-V1-2 and SciRE-V1-3, respectively, the details are available in Appendix H. Meanwhile, we consider using the sigmoid-type time trajectory only when NFE is less than 15. The superior of SciRE-V1 is particularly evident in its ability to generate high-quality samples with 2.42 FID in just 20 NFE, as shown in (b) of Figure 7. Furthermore, supported by several experimental validations, SciRE-V1 achieves 2.40 FID in just 100 NFE, which attains a new SOTA value under the VP-deep model (Song et al., 2021c) that we used.

![](images/472c30905d391564cbab4b0461685c45337a1deb7dbbd5b7129bde059b9d9a05.jpg)  
Figure 5: Compare the generation results of the RD-based methods (Solvers: SciRE-V1-2, SciREI-2) and the baseline method (Solver: DPM-2) using 6-36 sampling steps with the uniform time trajectory and identical settings, on pre-trained models with ImageNet  $128 \times 128$  and LSUN bedroom  $256 \times 256$ .

![](images/4c199a78249b0cc2b7eabf82e485915fbd05c62fb860def36e49e1a03a211eb4.jpg)  
(a) DPM-Solver++ (multistep).

![](images/02cc7f341e13f826aff51d14e41bde21ec1cd5e625c3560950deb0e93f32ace7.jpg)  
(b) SciRE-V1-2m, Algorithm 4.

![](images/4145027ca26f87600c24d3c91074f73cf32aff8b8cd00c90a00ed96ca709c9c7.jpg)  
(c) SciRE-V2-2m, Algorithm 7.

![](images/6caa88c99d936f085abfb5defa8ccaf1c3bb038201adf4170ae99ccef7ffe54e.jpg)  
Figure 6: Random samples of Stable-Diffusion, using only 6 NFE and text prompt "A beautiful mansion beside a waterfall in the woods, by josef thoma, matte painting, trending on artstation HQ".  
(a) CIFAR-10 (discrete)  
Figure 7: The comparative diagram of FID  $\downarrow$  of different training-free sampling methods on the CIFAR-10 and CelebA  $64\times 64$  datasets. In these three cases, our samplers reach SOTA.

![](images/43a206b42c0fa61b5bbfda0592a8587641b841b51bd8f03340e3a1e2160439a4.jpg)  
(b) CIFAR-10 (continuous)

![](images/4a100f72a921a760f33a2284f7b1af483b43962529a4dc3e3912e8ae5272e2ea.jpg)  
(c) CelebA  $64 \times 64$  (discrete)

# 6 CONCLUSIONS

In this work, we introduce the recursive difference (RD) method to calculate the derivative of the score function evaluations in the realm of diffusion models. By applying the RD method to the truncated Taylor expansion of the score-integrand, we propose the SciRE-Solver with the convergence order guarantee to accelerate the sampling process of DMs. The effectiveness of the RD method in evaluating the derivative of the score function in regular diffusion modes has been confirmed

<table><tr><td>Sampling method \NFE</td><td>12</td><td>15</td><td>20</td><td>50</td><td>200</td><td>1000</td></tr><tr><td colspan="7">CIFAR-10 (discrete-time model (Ho et al., 2020), linear noise schedule)</td></tr><tr><td>DDPM</td><td>246.3</td><td>197.6</td><td>137.3</td><td>32.6</td><td>4.03</td><td>3.16</td></tr><tr><td>DDIM</td><td>11.02</td><td>8.92</td><td>6.94</td><td>4.73</td><td>4.07</td><td>3.95</td></tr><tr><td>Analytic-DDIM</td><td>11.68</td><td>9.16</td><td>7.20</td><td>4.28</td><td>3.60</td><td>3.86</td></tr><tr><td>tAB3-DEIS</td><td colspan="2">7.12(10NFE)</td><td>4.53</td><td>3.78</td><td>\</td><td>\</td></tr><tr><td>DPM-Solver-2</td><td>6.15</td><td>†5.23</td><td>3.95</td><td>3.50</td><td>3.46</td><td>3.46</td></tr><tr><td>DPM-Solver-3</td><td>8.20</td><td>5.21</td><td>†3.81</td><td>†3.49</td><td>†3.45</td><td>†3.45</td></tr><tr><td>F-PNDM</td><td colspan="2">7.03(10NFE)</td><td>4.61</td><td>3.68</td><td>3.47</td><td>3.26</td></tr><tr><td>ERA-Solver</td><td>4.38</td><td>3.86</td><td>3.79</td><td>3.42</td><td colspan="2">3.51(100NFE)</td></tr><tr><td>SciRE-V1-2 (ours)</td><td>4.41</td><td>†4.09</td><td>3.67</td><td>3.28</td><td colspan="2">3.26(100NFE)</td></tr><tr><td>SciRE-V1-3 (ours)</td><td>5.00</td><td>4.12</td><td>†3.80</td><td>†3.23</td><td colspan="2">3.15 (84NFE)</td></tr><tr><td colspan="7">CIFAR-10 (VP deep continuous-time model (Song et al., 2021c))</td></tr><tr><td>DPM-Solver-2</td><td>4.88</td><td>†4.23</td><td>3.26</td><td>2.69</td><td>2.60</td><td>2.59</td></tr><tr><td>DPM-Solver-3</td><td>5.53</td><td>3.55</td><td>†2.90</td><td>†2.65</td><td>†2.62</td><td>†2.62</td></tr><tr><td>DPM-Solver-fast</td><td>4.93</td><td>3.35</td><td>2.87</td><td>\</td><td>\</td><td>\</td></tr><tr><td>tAB3-DEIS</td><td>\</td><td>3.37</td><td>2.86</td><td>2.57</td><td>\</td><td>\</td></tr><tr><td>SciRE-V1-2 (ours)</td><td>4.33</td><td>†3.84</td><td>3.03</td><td>2.57</td><td colspan="2">2.48 (100NFE)</td></tr><tr><td>SciRE-V1-3 (ours)</td><td>3.48</td><td>3.06</td><td>†2.68</td><td>†2.54</td><td colspan="2">†2.44 (100NFE)</td></tr><tr><td>SciRE-V1-agile (ours)</td><td>4.80</td><td>3.47</td><td>2.42</td><td>2.52</td><td colspan="2">2.40 (100NFE)</td></tr><tr><td colspan="7">CIFAR-10 (edm (Karras et al., 2022))</td></tr><tr><td>EDM-Heun</td><td>7.28</td><td>†4.47</td><td>2.38</td><td>1.83</td><td colspan="2">1.84 (100NFE)</td></tr><tr><td>SciRE-V1-2</td><td>2.29</td><td>†2.16</td><td>1.94</td><td>1.79</td><td colspan="2">1.76 (100NFE)</td></tr><tr><td colspan="7">CelebA 64×64 (discrete-time model (Song et al., 2021a), linear noise schedule)</td></tr><tr><td>Sampling method \NFE</td><td>10</td><td>12</td><td>15</td><td>20</td><td>50</td><td>1000</td></tr><tr><td>DDIM</td><td>10.85</td><td>9.99</td><td>7.78</td><td>6.64</td><td>5.23</td><td>4.88</td></tr><tr><td>DPM-Solver</td><td>5.83</td><td>3.71</td><td>3.05</td><td>2.82</td><td>2.71</td><td>(36NFE)</td></tr><tr><td>F-PNDM</td><td>7.71</td><td>\</td><td>\</td><td>5.51</td><td>3.34</td><td>2.71</td></tr><tr><td>tAB3-DEIS</td><td>6.95</td><td>\</td><td>\</td><td>3.41</td><td>2.95</td><td>\</td></tr><tr><td>SciRE-V1-2 (ours)</td><td>4.91</td><td>3.91</td><td>†3.38</td><td>2.56</td><td>2.30</td><td>-</td></tr><tr><td>SciRE-V1-3 (ours)</td><td>†9.72</td><td>4.07</td><td>2.53</td><td>†2.17</td><td>†2.02</td><td>-</td></tr></table>

Table 1: Generation quality measured by FID ↓ of different sampling methods for DMs on CIFAR-10 and CelebA 64×64. In this Table, we compare the best FID reported in existing literature with the FID achieved by our proposed SciRE-V1 at the same NFE. The bold black represents the best result obtained under the same NFE (column). The results with  $\dagger$  means the actual NFE is smaller than the given NFE because the given NFE cannot be divided by 2 or 3. Some results are missing in their original papers, which are replaced by “ $\backslash$ ”. Here, we used the same time trajectory scheme to evaluate the results of SciRE-V1 on CIFAR-10 and CelebA 64×64 datasets with discrete models. The setting of continuous-time on CIFAR-10 are described in Section 5.2. More comparisons and additional details are shown in Appendix H.

through comparative experiments involving FID and generated samples. These experiments were conducted for ablation comparisons with both the finite-based difference algorithm and the popular DPM-Solver-2 algorithm. SciRE-Solver (versions: V1 and V2) is a new type of algorithm that provides an alternative sampling scheme for accelerating diffusion models. Numerical experiments indicate that SciRE-Solver not only can generate high-quality samples across various datasets using fewer-steps but also, using a small NFEs demonstrates promising potential to surpass the FID achieved by some pre-trained models in their original papers using no fewer than 1000 NFEs.

![](images/ef4c2cc7cc2cb35b91ffcba2cfe09467f78f525ad8f9cd8d3c934bc5896869ef.jpg)  
Figure 8: Random samples of Stable-Diffusion by SciRE-V1 and SciRE-V2, using varying NFEs and the text prompt "a girl face in Disney style, physically-based rendering, ultimate painting, UHD".

# REFERENCES

Kendall Atkinson, Weimin Han, and David E Stewart. Numerical solution of ordinary differential equations. John Wiley & Sons, 2011.  
Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=0xiJLKH-ufZ.  
Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=NsMLjcFa080.  
Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=31ge@p5o-M-.  
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780-8794, 2021.  
Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with critically-damped Langevin diffusion. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=CzceR82CYc.  
Jack K Hale and Sjoerd M Verduyn Lunel. Introduction to functional differential equations, volume 99. Springer Science & Business Media, 2013.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020.  
Jonathan Ho, William Chan, Chitwan Sahara, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.  
Marlis Hochbruck and Alexander Ostermann. Explicit exponential runge-kutta methods for semilinear parabolic problems. SIAM Journal on Numerical Analysis, 43(3):1069-1090, 2005.  
Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021.  
Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565-26577, 2022.  
Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696-21707, 2021.  
Shengmeng Li, Luping Liu, Zenghao Chai, Runnan Li, and Xu Tan. Era-solver: Error-robust adams solver for fast sampling of diffusion probabilistic models. arXiv preprint arXiv:2301.12935, 2023.  
Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, and Zhou Zhao. Diffsinger: Singing voice synthesis via shallow diffusion mechanism. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 11020-11028, 2022a.  
Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2022b. URL https://openreview.net/forum?id=PlKWVd2yBkY.

Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based diffusion odes by high order denoising score matching. In International Conference on Machine Learning, pp. 14429-14460. PMLR, 2022a.  
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 2022b.  
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022c.  
Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.  
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= aBsCjcPu_tE.  
Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14297-14306, 2023.  
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162-8171. PMLR, 2021.  
Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pp. 16784-16804. PMLR, 2022.  
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684-10695, 2022.  
Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2021.  
Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=TIdIXIpzhoI.  
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pp. 2256-2265. PMLR, 2015.  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id=St1giarCHLP.  
Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in Neural Information Processing Systems, 34:1415-1428, 2021b.  
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021c. URL https://openreview.net/forum?id=PxTIG12RRHS.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.  
Endre Süli. Numerical solution of ordinary differential equations. Mathematical Institute, University of Oxford, 2010.  
Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. Advances in Neural Information Processing Systems, 34:11287-11302, 2021a.  
Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021b. URL https://openreview.net/forum?id=P9TYG@j-wtG.  
Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion GANs. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=JprM@p-q@Co.  
Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.  
Qinsheng Zhang and Yongxin Chen. Diffusion normalizing flow. Advances in Neural Information Processing Systems, 34:16280-16291, 2021.  
Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=Loek7hfb46P.  
Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609-3623, 2022.  
Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867, 2023.  
Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Improved techniques for maximum likelihood estimation for diffusion odes. arXiv preprint arXiv:2305.03935, 2023.
