# Elucidating the Design Space of Diffusion-Based Generative Models

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of an existing ImageNet-64 model from 2.07 to near-SOTA 1.55.

# 1 Introduction

Diffusion-based generative models have emerged as a powerful new framework for neural image synthesis, in both unconditional [15, 32, 41] and conditional [16, 31, 32, 34, 35, 36, 37, 41] settings, even surpassing the quality of GANs [12] in certain situations [9]. They are also rapidly finding use in other domains such as audio [26, 33] and video [18] generation, image segmentation [4, 47] and language translation [30]. As such, there is great interest in applying these models and improving them further in terms of image/distribution quality, training cost, and generation speed.  
The literature on these models is dense on theory, and derivations of sampling schedule, training dynamics, noise level parameterization, etc., tend to be based as directly as possible on theoretical frameworks, which ensures that the models are on a solid theoretical footing. However, this approach has a danger of obscuring the available design space—a proposed model may appear as a tightly coupled package where no individual component can be modified without breaking the entire system.  
As our first contribution, we take a look at the theory behind these models from a practical standpoint, focusing more on the "tangible" objects and algorithms that appear in the training and sampling phases, and less on the statistical processes from which they might be derived. The goal is to obtain better insights into how these components are linked together and what degrees of freedom are available in the design of the overall system. We focus on the broad class of models where a neural network is used to model the score [21] of a noise level dependent marginal distribution of the training data corrupted by Gaussian noise. Thus, our work is in the context of denoising score matching [44].  
Our second set of contributions concerns the sampling processes used to synthesize images using diffusion models. We identify the best-performing time discretization for sampling, apply a higher-order Runge-Kutta method for the sampling process, evaluate different sampler schedules, and analyze the usefulness of stochasticity in the sampling process. The result of these improvements is a significant drop in the number of sampling steps required during synthesis, and the improved sampler can be used as a drop-in replacement with several widely used diffusions models [32, 41].

The third set of contributions focuses on the training of the score-modeling neural network. While we continue to rely on the commonly used network architectures (DDPM [15], NCSN [40]), we provide the first principled analysis of the preconditioning of the networks' inputs, outputs, and loss functions in a diffusion model setting and derive best practices for improving the training dynamics. We also suggest an improved distribution of noise levels during training, and note that non-leaking augmentation [24]—typically used with GANs—is beneficial for diffusion models as well.

Taken together, our contributions enable significant improvements in result quality, e.g., leading to a record FID of 1.79 on CIFAR-10 [27]. With all key ingredients of the design space explicitly tabulated, we believe that our approach will allow easier innovation on the individual components, and thus enable more extensive and targeted exploration of the design space of diffusion models.

# 2 Expressing diffusion models in a common framework

Let us denote the data distribution by  $p_{\mathrm{data}}(\pmb{x})$ , with standard deviation  $\sigma_{\mathrm{data}}$ , and consider the family of mollified distributions  $p(\pmb{x};\sigma)$  obtained by adding i.i.d. Gaussian noise of standard deviation  $\sigma$  to the data. For  $\sigma_{\mathrm{max}} \gg \sigma_{\mathrm{data}}$ ,  $p(\pmb{x};\sigma_{\mathrm{max}})$  is practically indistinguishable from pure Gaussian noise. The idea of diffusion models is to randomly sample a noise image  $\pmb{x}_0 \sim \mathcal{N}(\mathbf{0},\sigma_{\mathrm{max}}^2\mathbf{I})$ , and sequentially denoise it into images  $\pmb{x}_i$  with noise levels  $\sigma_0 = \sigma_{\mathrm{max}} > \sigma_1 > \dots >\sigma_N = 0$  so that at each noise level  $\pmb{x}_i \sim p(\pmb{x}_i;\sigma_i)$ . The endpoint  $\pmb{x}_N$  of this process is thus distributed according to the data.

Song et al. [41] present a stochastic differential equation (SDE) that maintains the desired distribution  $p$  as sample  $x$  evolves over time. This allows the above process to be implemented using a stochastic solver that both removes and adds noise at each iteration. They also give a corresponding "probability flow" ordinary differential equation (ODE) where the only source of randomness is the initial noise image  $x_0$ . Contrary to the usual order of treatment, we begin by examining the ODE, as it offers a fruitful setting for analyzing sampling trajectories and their discretizations. The insights carry over to stochastic sampling, which we reintroduce as a generalization in Section 4.

ODE formulation. A probability flow ODE [41] continuously increases or reduces noise level of the image when moving forward or backward in time, respectively. To specify the ODE, we must first choose a schedule  $\sigma(t)$  that defines the desired noise level at time  $t$ . For example, setting  $\sigma(t) \propto \sqrt{t}$  is mathematically natural, as it corresponds to constant-speed heat diffusion [11]. However, we will show in Section 3 that the choice of schedule has major practical implications and should not be made on the basis of theoretical convenience.

The defining characteristic of the probability flow ODE is that evolving a sample  $\pmb{x}_a \sim p\big(\pmb{x}_a; \sigma(t_a)\big)$  from time  $t_a$  to  $t_b$  (either forward or backward in time) yields a sample  $\pmb{x}_b \sim p\big(\pmb{x}_b; \sigma(t_b)\big)$ . Following previous work [41], this requirement is satisfied (Appendix B) by

$$
\mathrm {d} \boldsymbol {x} = - \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm {d} t, \tag {1}
$$

where the dot denotes a time derivative.  $\nabla_{\pmb{x}}\log p(\pmb {x};\sigma)$  is the score function [21], a vector field that points towards higher density of data at a given noise level. Intuitively, an infinitesimal forward step of this ODE nudges the sample away from the data, at a rate that depends on the change in noise level. Equivalently, a backward step nudges the sample towards the data distribution.

Denoising score matching. The score function has the remarkable property that it does not depend on the generally intractable normalization constant of the underlying density function  $p(\pmb{x};\sigma)$  [21], and thus can be much easier to evaluate. Specifically, if  $D(\pmb{x};\sigma)$  is a denoiser function that minimizes the expected  $L_{2}$  denoising error for samples drawn from  $p_{\mathrm{data}}$  separately for every  $\sigma$ , i.e.,

$$
\mathbb {E} _ {\boldsymbol {y} \sim p _ {\text {d a t a}}} \mathbb {E} _ {\boldsymbol {n} \sim \mathcal {N} (\boldsymbol {0}, \sigma^ {2} \mathbf {I})} \| D (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \| _ {2} ^ {2}, \text {t h e n} \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma) = (D (\boldsymbol {x}; \sigma) - \boldsymbol {x}) / \sigma^ {2}, \tag {2.3}
$$

where  $\pmb{y}$  is a training image and  $\pmb{n}$  is noise. In this light, the score function isolates the noise component from the signal in  $\pmb{x}$ , and Eq. 1 amplifies (or diminishes) it over time. Figure 1 illustrates the behavior of ideal  $D$  in practice. The key observation in diffusion models is that  $D(\pmb{x};\sigma)$  can be implemented as a neural network  $D_{\theta}(\pmb{x};\sigma)$  trained according to Eq. 2. Note that  $D_{\theta}$  may include additional pre- and post-processing steps, such as scaling  $\pmb{x}$  to an appropriate dynamic range; we will return to such preconditioning in Section 5.

Time-dependent signal scaling. Some methods (see Appendix C) introduce an additional scale schedule  $s(t)$  and consider  $\pmb{x} = s(t)\hat{\pmb{x}}$  to be a scaled version of the original, non-scaled variable

![](images/d9b67daaa171158f023445902cda2aaa2032adad16a2e596a223901749836834.jpg)  
(a) Noisy images drawn from  $p(\pmb{x};\sigma)$

![](images/95c89fa11b2f117366ce03b440811f12b3e794e5b488d85927a6cdefeab1041f.jpg)  
Figure 1: Denoising score matching on CIFAR-10. (a) Images from the training set corrupted with varying levels of additive Gaussian noise. High levels of noise lead to oversaturated colors; we normalize the images for cleaner visualization. (b) Optimal denoising result from minimizing Eq. 2 analytically (Appendix B). With increasing noise level, the denoised image approaches dataset mean.  
(b) Ideal denoiser outputs  $D(\pmb{x};\sigma)$

Table 1: Specific design choices employed by different model families.  $N$  is the number of ODE solver iterations that we wish to execute during sampling. The corresponding sequence of time steps is  $\{t_0,t_1,\dots ,t_N\}$ , where  $t_N = 0$ . If the model was originally trained for specific choices of  $N$  and  $\{t_i\}$ , the originals are denoted by  $M$  and  $\{u_j\}$ , respectively. The denoiser is defined as  $D_{\theta}(\pmb {x};\sigma) = c_{\mathrm{skip}}(\sigma)\pmb {x} + c_{\mathrm{out}}(\sigma)F_{\theta}\big(c_{\mathrm{in}}(\sigma)\pmb {x};c_{\mathrm{noise}}(\sigma)\big);F_{\theta}$  represents the raw neural network layers.  

<table><tr><td colspan="2"></td><td>VP [41]</td><td>VE [41]</td><td>iDDPM [32] + DDIM [39]</td><td>Ours</td></tr><tr><td colspan="2">Sampling (Section 3)</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">ODE solver</td><td>Euler</td><td>Euler</td><td>Euler</td><td>2ndorder Heun</td></tr><tr><td rowspan="2">Time steps</td><td rowspan="2">ti&lt;N</td><td rowspan="2">1 + i/N-1(εs-1)</td><td rowspan="2">σ2max(σ2min/σ2max)iN-1</td><td>u[j0+M-1-j0i+1/2], where uM=0</td><td rowspan="2">(σmax1/p)+ i/N-1(σmin1/p-σmax1/p)ρ</td></tr><tr><td>uj-1=√u2j+1/ max(αj-1/αj,C1)-1</td></tr><tr><td>Schedule</td><td>σ(t)</td><td>√e1/2βdt2+βmint-1</td><td>√t</td><td>t</td><td>t</td></tr><tr><td>Scaling</td><td>s(t)</td><td>1/√e1/2βdt2+βmint</td><td>1</td><td>1</td><td>1</td></tr><tr><td colspan="2">Network and preconditioning (Section 5)</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Architecture of Fθ</td><td>DDPM++</td><td>NCSN++</td><td>DDPM</td><td>(any)</td></tr><tr><td>Skip scaling</td><td>cskip(σ)</td><td>1</td><td>1</td><td>1</td><td>σ2data/ (σ2+σdata2)</td></tr><tr><td>Output scaling</td><td>cout(σ)</td><td>-σ</td><td>σ</td><td>-σ</td><td>σ·σdata/√σ2data+σ2</td></tr><tr><td>Input scaling</td><td>cin(σ)</td><td>1/√σ2+1</td><td>1</td><td>1/√σ2+1</td><td>1/√σ2+σ2data</td></tr><tr><td>Noise cond.</td><td>cnoise(σ)</td><td>(M-1) σ-1(σ)</td><td>ln(1/2σ)</td><td>M-1-arg minj|uj-σ|</td><td>1/4 ln(σ)</td></tr><tr><td colspan="2">Training (Section 5)</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Noise distribution</td><td>σ-1(σ) ~ U(εt,1)</td><td>ln(σ) ~ U(ln(σmin), ln(σmax))</td><td>σ =uj, j~U{0,M-1}</td><td>ln(σ) ~ N(Pmean,Pstd2)</td></tr><tr><td>Loss weighting</td><td>λ(σ)</td><td>1/σ2</td><td>1/σ2</td><td>1/σ2 (note:* )</td><td>(σ2+σdata2)/(σ·σdata2)</td></tr><tr><td rowspan="3" colspan="2">Parameters</td><td>βd=19.9, βmin=0.1</td><td>σmin=0.02</td><td>a_j=sin^2(π/2 M(C2+1))</td><td>σmin=0.002, σmax=80</td></tr><tr><td>εs=10-3, εt=10-5</td><td>σmax=100</td><td>C1=0.001, C2=0.008</td><td>σdata=0.5, ρ=7</td></tr><tr><td>M=1000</td><td></td><td>M=1000, j0=8†</td><td>Pmean=-1.2, Pstd=1.2</td></tr></table>

* iDDPM also employs a second loss term  ${L}_{\mathrm{{v1b}}}$  † In our tests,  ${j}_{0} = 8$  yielded better FID than  ${j}_{0} = 0$  used by iDDPM

$\hat{x}$ . This changes the time-dependent probability density, and consequently also the ODE solution trajectories. The resulting ODE is a generalization of Eq. 1:

$$
\mathrm {d} \boldsymbol {x} = \left[ \dot {s} (t) \boldsymbol {x} / s (t) - s (t) ^ {2} \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t)) \right] \mathrm {d} t. \tag {4}
$$

Note that we explicitly undo the scaling of  $\pmb{x}$  when evaluating the score function to keep the definition of  $p(\pmb{x};\sigma)$  independent of  $s(t)$ .

Solution by discretization. The ODE to be solved is obtained by substituting Eq. 3 into Eq. 4 to define the point-wise gradient, and the solution can be found by numerical integration, i.e., taking finite steps over discrete time intervals. This requires choosing both the integration scheme (e.g., Euler or a variant of Runge-Kutta), as well as the discrete sampling times  $\{t_0, t_1, \ldots, t_N\}$ . Many prior works rely on Euler's method, but we show in Section 3 that a  $2^{\mathrm{nd}}$  order solver offers a better computational tradeoff. For brevity, we do not provide a separate pseudocode for Euler's method applied to our ODE here, but it can be extracted from Algorithm 1 by omitting lines 6-8.

Putting it together. Table 1 presents formulas for reproducing deterministic variants of three earlier methods in our framework. These methods were chosen because they are widely used and achieve state-of-the-art performance, but also because they were derived from different theoretical foundations. Some of our formulas appear quite different from the original papers as indirection and recursion have been removed; see Appendix C for details. The main purpose of this reframing is to

![](images/a83a11b9592f89959705c93386c37c5008c462a4a2a70ac57669126dc70c3b3a.jpg)  
(a) Uncond. CIFAR-10, VP ODE

![](images/2af8adc7ce3138cc9ad45303b730167699be7119c367d80141f54889c8d91d0e.jpg)  
Figure 2: Comparison of deterministic sampling methods using three pre-trained models. For each curve, the dot indicates the lowest NFE whose FID is within  $3\%$  of the lowest observed FID.

![](images/5bd5ec64b2efdfcdf0d6267b5152a25fd6c82bfd47abe0aebb914cd1abd93769.jpg)  
(b) Uncond. CIFAR-10, VE ODE  
(c) Class-cond. ImageNet-64, DDIM

bring into light all the independent components that often appear tangled together in previous work. In our framework, there are no implicit dependencies between the components—any choices (within reason) for the individual formulas will, in principle, lead to a functioning model.

# 3 Improvements to deterministic sampling

Our hypothesis is that the choices related to the sampling process are largely independent of the other components, such as network architecture and training details. In other words, the training procedure of  $D_{\theta}$  should not dictate  $\sigma(t), s(t)$ , and  $\{t_i\}$ , nor vice versa; from the viewpoint of the sampler,  $D_{\theta}$  is simply a black box [45, 46]. We test this by evaluating different samplers on three pre-trained models, each representing a different theoretical framework and model family. We first measure baseline results for these models using their original sampler implementations, and then bring these samplers into our unified framework using the formulas in Table 1, followed by our improvements.

We evaluate the "DDPM++ cont. (VP)" and "NCSN++ cont. (VE)" models by Song et al. [41] trained on unconditional CIFAR-10 [27] at  $32 \times 32$ , corresponding to the variance preserving (VP) and variance exploding (VE) formulations [41], originally inspired by DDPM [15] and SMLD [40]. We also evaluate the "ADM (dropout)" model by Dhariwal and Nichol [9] trained on class-conditional ImageNet [8] at  $64 \times 64$ , corresponding to the improved DDPM (iDDPM) formulation [32]. This model was trained using a discrete set of  $M = 1000$  noise levels. Further details are given in Appendix C.

We evaluate the result quality in terms of Fréchet inception distance (FID) [14] computed between 50,000 generated images and all available real images. Figure 2 shows FID as a function of neural function evaluations (NFE), i.e., how many times  $D_{\theta}$  is evaluated to produce a single image. Given that the sampling process is dominated entirely by the cost of  $D_{\theta}$ , improvements in NFE translate directly to sampling speed. The original deterministic samplers are shown in blue, and the reimplementations of these methods in our unified framework (orange) yield similar but consistently better results. The differences are explained by certain oversights in the original implementations as well as our more careful treatment of discrete noise levels in the case of DDIM; see Appendix C. Note that our reimplementations are fully specified by Algorithm 1 and Table 1, even though the original codebases are structured very differently from each other.

Discretization and higher-order integrators. Solving an ODE numerically is necessarily an approximation of following the true solution trajectory. At each step, the solver introduces truncation error that accumulates over the course of  $N$  steps. The local error generally scales superlinearly with respect to step size, and thus increasing  $N$  improves the accuracy of the solution.

The commonly used Euler's method is a first order ODE solver with  $\mathcal{O}(h^2)$  local error with respect to step size  $h$ . Higher-order Runge-Kutta methods [42] scale more favorably but require multiple evaluations of  $D_{\theta}$  per step. Through extensive tests, we have found Heun's  $2^{\mathrm{nd}}$  order method [2] (a.k.a. improved Euler, trapezoidal rule)—previously explored in the context of diffusion models by Jolicoeur-Martineau et al. [23]—to provide an excellent tradeoff between truncation error and NFE. As illustrated in Algorithm 1, it introduces an additional correction step for  $\pmb{x}_{i + 1}$  to account for change in  $\mathrm{d}\pmb {x} / \mathrm{d}t$  between  $t_i$  and  $t_{i + 1}$ . This correction leads to  $\mathcal{O}(h^3)$  local error at the cost of one additional evaluation of  $D_{\theta}$  per step. Note that stepping to  $\sigma = 0$  would result in a division by zero, so we revert to Euler's method in this case. We discuss the general family of  $2^{\mathrm{nd}}$  order solvers in Appendix D.

The time steps  $\{t_i\}$  determine how the step sizes and thus truncation errors are distributed between different noise levels. We provide a detailed analysis in Appendix D, concluding that the step size should decrease monotonically with decreasing  $\sigma$  and it does not need to vary on a per-sample basis.

Algorithm 1 Deterministic sampling using Heun's  $2^{\mathrm{nd}}$  order method with arbitrary  $\sigma(t)$  and  $s(t)$ .  
1: procedure HEUNNSAMPLER  $(D_{\theta}(\pmb {x};\sigma)$ $\sigma (t),s(t),t_{i\in \{0,\dots ,N\}})$    
2: sample  $\pmb {x}_0\sim \mathcal{N}\big(\pmb {0},\sigma^2 (t_0)s^2 (t_0)\textbf{I}\big)$  Generate initial sample at  $t_0$    
3: for  $i\in \{0,\ldots ,N - 1\}$  do Solve Eq.4 over  $N$  time steps   
4:  $\pmb {d}_i\gets \left(\frac{\dot{\sigma}(t_i)}{\sigma(t_i)} +\frac{\dot{s}(t_i)}{s(t_i)}\right)\pmb {x}_i - \frac{\dot{\sigma}(t_i)s(t_i)}{\sigma(t_i)} D_\theta \left(\frac{\pmb{x}_i}{s(t_i)};\sigma(t_i)\right)$  Evaluate d/dt at ti   
5:  $\pmb{x}_{i + 1}\gets \pmb {x}_i + (t_{i + 1} - t_i)\pmb {d}_i$  Take Euler step from  $t_i$  to  $t_{i + 1}$    
6: if  $\sigma (t_{i + 1})\neq 0$  then Apply  $2^{\mathrm{nd}}$  order correction unless  $\sigma$  goes to zero   
7:  $\pmb{d}_i^\prime \gets \left(\frac{\dot{\sigma}(t_{i + 1})}{\sigma(t_{i + 1})} +\frac{\dot{s}(t_{i + 1})}{s(t_{i + 1})}\right)\pmb{x}_{i + 1} - \frac{\dot{\sigma}(t_{i + 1})s(t_{i + 1})}{\sigma(t_{i + 1})} D_\theta \left(\frac{\pmb{x}_{i + 1}}{s(t_{i + 1})};\sigma (t_{i + 1})\right)$  Evaluate.d/dt at  $t_{i + 1}$    
8:  $\pmb{x}_{i + 1}\gets \pmb {x}_i + (t_{i + 1} - t_i)\left(\frac{1}{2}\pmb {d}_i + \frac{1}{2}\pmb {d}_i'\right)$  Explicit trapezoidal rule at  $t_{i + 1}$    
9: return  $x_N$  Return noise-free sample at  $t_N$

![](images/a81c5210b4f1bc48e27736738a04934819ba972cab01a09ca6cf67f0591c521f.jpg)  
(a) Variance preserving ODE [41]

![](images/0ac617196293aaee29995f0b6c8915dab0acc29f0297f5dc2ca2b1bde3e52e43.jpg)  
Figure 3: A sketch of ODE curvature in 1D where  $p_{\mathrm{data}}$  is two Dirac peaks at  $x = \pm 1$ . Horizontal  $t$  axis is chosen to show  $\sigma \in [0,25]$  in each plot, with insets showing  $\sigma \in [0,1]$  near the data. Example local gradients are shown with black arrows. (a) Variance preserving ODE of Song et al. [41] has solution trajectories that flatten out to horizontal lines at large  $\sigma$ . Local gradients start pointing towards data only at small  $\sigma$ . (b) Variance exploding variant has extreme curvature near data and the solution trajectories are curved everywhere. (c) With the schedule used by DDIM [39] and us, as  $\sigma$  increases the solution trajectories approach straight lines that point towards the mean of data. As  $\sigma \to 0$ , the trajectories become linear and point towards the data manifold.

![](images/0d0cbc9d9e9d38b02ebb31170fdb560c71cc2d40f6833a0abc3b3877df4138c7.jpg)  
(b) Variance exploding ODE [41]  
(c) DDIM [39] / Our ODE

We adopt a parameterized scheme where the time steps are defined according to a sequence of noise levels  $\{\sigma_i\}$ , i.e.,  $t_i = \sigma^{-1}(\sigma_i)$ . We set  $\sigma_{i < N} = (Ai + B)^\rho$  and select the constants  $A$  and  $B$  so that  $\sigma_0 = \sigma_{\mathrm{max}}$  and  $\sigma_{N - 1} = \sigma_{\mathrm{min}}$ , which gives

$$
\sigma_ {i <   N} = \left(\sigma_ {\max } ^ {\frac {1}{\rho}} + \frac {i}{N - 1} \left(\sigma_ {\min } ^ {\frac {1}{\rho}} - \sigma_ {\max } ^ {\frac {1}{\rho}}\right)\right) ^ {\rho} \quad \text {a n d} \quad \sigma_ {N} = 0. \tag {5}
$$

Here  $\rho$  controls how much the steps near  $\sigma_{\mathrm{min}}$  are shortened at the expense of longer steps near  $\sigma_{\mathrm{max}}$ . Our analysis in Appendix D shows that setting  $\rho = 3$  nearly equalizes the truncation error at each step, but that  $\rho$  in range of 5 to 10 performs much better for sampling images. This suggests that errors near  $\sigma_{\mathrm{min}}$  have a large impact. We set  $\rho = 7$  for the remainder of this paper. Results for Heun's method and Eq. 5 are shown as the green curves in Figure 2. We observe consistent improvement in all cases: Heun's method reaches the same FID as Euler's method with considerably lower NFE.

Trajectory curvature and noise schedule. The shape of the ODE solution trajectories is defined by functions  $\sigma(t)$  and  $s(t)$ . The choice of these functions offers a way to reduce the truncation errors discussed above, as their magnitude can be expected to scale proportional to the curvature of  $\mathrm{d}\pmb{x} / \mathrm{d}t$ . We argue that the best choice for these functions is  $\sigma(t) = t$  and  $s(t) = 1$ , which is also the choice made in DDIM [39]. With this choice, the ODE of Eq. 4 simplifies to  $\mathrm{d}\pmb{x} / \mathrm{d}t = (\pmb{x} - D(\pmb{x}; t)) / t$ .

An immediate consequence is that at any  $\pmb{x}$  and  $t$ , a single Euler step to  $t = 0$  yields the denoised image  $D_{\theta}(\pmb{x};t)$ . The tangent of the solution trajectory therefore always points towards the denoiser output. This can be expected to change only slowly with the noise level, which corresponds to largely linear solution trajectories. The 1D ODE sketch of Figure 3c supports this intuition; the solution trajectories approach linear at both large and small noise levels, and have substantial curvature in only a small region in between. The same effect can be seen with real data in Figure 1b, where the change between different denoiser targets occurs in a relatively narrow  $\sigma$  range. With the advocated schedule, this corresponds to high ODE curvature being limited to this same range.

Algorithm 2 Our stochastic sampler with  $\sigma (t) = t$  and  $s(t) = 1$  
1: procedure STOCHASTIC SAMPLER  $(D_{\theta}(\pmb{x};\sigma), t_{i\in \{0,\dots,N\}}, \gamma_{i\in \{0,\dots,N-1\}}, S_{\text{noise}})$   
2: sample  $\pmb{x}_0 \sim \mathcal{N}(\pmb{0}, t_0^2\mathbf{I})$   
3: for  $i \in \{0,\dots,N-1\}$  do  
4: sample  $\pmb{\epsilon}_i \sim \mathcal{N}(\pmb{0}, S_{\text{noise}}^2\mathbf{I})$   
5:  $\hat{\pmb{t}}_i \gets t_i + \gamma_it_i$   
6:  $\hat{\pmb{x}}_i \gets \pmb{x}_i + \sqrt{\hat{t}_i^2 - t_i^2}\pmb{\epsilon}_i$   
7:  $\pmb{d}_i \gets (\hat{\pmb{x}}_i - D_\theta(\hat{\pmb{x}}_i; \hat{t}_i)) / \hat{t}_i$   
8:  $\pmb{x}_{i+1} \gets \hat{\pmb{x}}_i + (t_{i+1} - \hat{t}_i)\pmb{d}_i$   
9: if  $t_{i+1} \neq 0$  then  
10:  $\pmb{d}_i' \gets (\pmb{x}_{i+1} - D_\theta(\pmb{x}_{i+1}; t_{i+1})) / t_{i+1}$   
11:  $\pmb{x}_{i+1} \gets \hat{\pmb{x}}_i + (t_{i+1} - \hat{t}_i)\left(\frac{1}{2}\pmb{d}_i + \frac{1}{2}\pmb{d}_i'\right)$   
12: return  $\pmb{x}_N$

The effect of setting  $\sigma(t) = t$  and  $s(t) = 1$  is shown as the red curves in Figure 2. As DDIM already employs these same choices, the red curve is identical to the green one for ImageNet-64. However, VP and VE benefit considerably from switching away from their original schedules.

Discussion. The choices that we made in this section to improve deterministic sampling are summarized in the Sampling part of Table 1. Together, they reduce the NFE needed to reach high-quality results by a large factor:  $7.3 \times$  for VP,  $300 \times$  for VE, and  $3.2 \times$  for DDIM, corresponding to the highlighted NFE values in Figure 2. In practice, we can generate 26.3 high-quality CIFAR-10 images per second on a single NVIDIA V100. The consistency of improvements corroborates our hypothesis that the sampling process is orthogonal to how each model was originally trained. As further validation, we show results for the adaptive RK45 method [10] using our schedule as the dashed black curves in Figure 2; the cost of this sophisticated ODE solver outweighs its benefits.

# 4 Stochastic sampling

Deterministic sampling offers many benefits, e.g., the ability to turn real images into their corresponding latent representations by inverting the ODE. However, it tends to lead to worse output quality [39, 41] than stochastic sampling that injects fresh noise into the image in each step. Given that ODEs and SDEs recover the same distributions in theory, what exactly is the role of stochasticity?

Background. The SDEs of Song et al. [41] can be generalized [19, 48] as a sum of the probability flow ODE of Eq. 1 and a varying-rate Langevin diffusion SDE [13]:

$\mathrm{d}\pmb{x}_{\pm} = \underbrace{-\dot{\sigma}(t)\sigma(t)\nabla_{\pmb{x}}\log p(\pmb{x};\sigma(t))\mathrm{d}t}_{\text{probability flow ODE (Eq. 1)}}\pm \underbrace{\beta(t)\sigma(t)^2\nabla_{\pmb{x}}\log p(\pmb{x};\sigma(t))\mathrm{d}t}_{\text{deterministic noise decay}} + \underbrace{\sqrt{2\beta(t)}\sigma(t)\mathrm{d}\omega_t}_{\text{noise injection}}$  (6)

where  $\mathrm{d}\omega_{t}$  is the standard Wiener process.  $\mathrm{d}\pmb{x}_{+}$  and  $\mathrm{d}\pmb{x}_{-}$  are now separate SDEs for moving forward and backward in time, related by the time reversal formula of Anderson [1]. The Langevin term can further be seen as a combination of a deterministic score-based denoising term and a stochastic noise injection term, whose net noise level contributions cancel out. As such,  $\beta (t)$  effectively expresses the relative rate at which existing noise is replaced with new noise. The SDEs of Song et al. [41] are recovered with the choice  $\beta (t) = \dot{\sigma} (t) / \sigma (t)$ , whereby the score vanishes from the forward SDE.

This perspective reveals why stochasticity is helpful in practice: The implicit Langevin diffusion drives the sample towards the desired marginal distribution at a given time, actively correcting for any errors made in earlier sampling steps. On the other hand, approximating the Langevin term with discrete SDE solver steps introduces error in itself. Previous results [3, 23, 39, 41] suggest that non-zero  $\beta(t)$  is helpful, but as far as we can tell, the implicit choice for  $\beta(t)$  in Song et al. [41] enjoys no special properties. Hence, the optimal amount of stochasticity should be determined empirically.

Our stochastic sampler. We propose a stochastic sampler that combines the existing higher-order ODE integrator with explicit Langevin-like "churn" of adding and removing noise. A pseudocode is given in Algorithm 2. At each step  $i$ , given the sample  $\pmb{x}_i$  at noise level  $t_i (= \sigma(t_i))$ , we perform two

![](images/822245badaa14bab22d133188acd16428f4cf5321f8f05e80f38a40e3588d03a.jpg)  
(a) Uncond. CIFAR-10, VP

![](images/06cf3e59b5310adb63b28a167fdefc014c62a3ef9e3bc6b6cecb0b209792902d.jpg)  
Figure 4: Evaluation of our stochastic sampler (Algorithm 2). The purple curve corresponds to optimal choices for  $\{S_{\mathrm{churn}}, S_{\mathrm{tmin}}, S_{\mathrm{tmax}}, S_{\mathrm{noise}}\}$ ; orange, blue, and green correspond to disabling the effects of  $S_{\mathrm{tmin,tmax}}$  and/or  $S_{\mathrm{noise}}$ . The red curves show reference results for our deterministic sampler (Algorithm 1), equivalent to setting  $S_{\mathrm{churn}} = 0$ . The dashed black curves correspond to the original stochastic samplers from previous work: Euler-Maruyama [41] for VP, predictor-corrector [41] for VE, and iDDPM [32] for ImageNet-64. The dots indicate lowest observed FID.  
(b) Uncond. CIFAR-10, VE

![](images/59d876456ab1e6c988a19b6b0df0931d5dcb268684bf40c4702d62b3bb36e36b.jpg)  
(c) Class-cond. ImageNet-64

sub-steps. First, we add noise to the sample according to a factor  $\gamma_{i} \geq 0$  to reach a higher noise level  $\hat{t}_{i} = t_{i} + \gamma_{i} t_{i}$ . Second, from the increased-noise sample  $\hat{\pmb{x}}_{i}$ , we solve the ODE backward from  $\hat{t}_{i}$  to  $t_{i+1}$  with a single step. This yields a sample  $\pmb{x}_{i+1}$  with noise level  $t_{i+1}$ , and the iteration continues. We stress that this is not a general-purpose SDE solver, but a sampling procedure tailored for the specific problem. Its correctness stems from the alternation of two sub-steps that each maintain the correct distribution (up to truncation error in the ODE step).

Practical considerations. We have observed (see Appendix E) that excessive Langevin-like addition and removal of noise results in gradual loss of detail in the generated images with all datasets and denoiser networks. There is also a drift toward oversaturated colors at very low and high noise levels. We suspect that practical denoisers induce a slightly non-conservative vector field in Eq. 3, violating the premises of Langevin diffusion and causing these detrimental effects. Notably, our experiments with analytical denoisers (such as the one in Figure 1b) have not shown such degradation.

If the degradation is caused by flaws in  $D_{\theta}(\pmb{x}; \sigma)$ , they can only be remedied using heuristic means during sampling. We address the drift toward oversaturated colors by only enabling stochasticity within a specific range of noise levels  $t_i \in [S_{\mathrm{tmin}}, S_{\mathrm{tmax}}]$ . For these noise levels, we define  $\gamma_i = S_{\mathrm{churn}} / N$ , where  $S_{\mathrm{churn}}$  controls the overall amount of stochasticity. We further clamp  $\gamma_i$  to never introduce more new noise than what is already present in the image. Finally, we have found that the loss of detail can be partially counteracted by setting  $S_{\mathrm{noise}}$  slightly above 1 to inflate the standard deviation for the newly added noise. This suggests that a major component of the hypothesized non-conservativity of  $D_{\theta}(\pmb{x}; \sigma)$  is a tendency to remove slightly too much noise—most likely due to regression toward the mean that can be expected to happen with any  $L_2$ -trained denoiser [28].

Evaluation. Figure 4 shows that our stochastic sampler outperforms previous samplers [32, 41] by a significant margin, especially at low step counts. In particular, through sampler improvements alone, we are able to bring the ImageNet-64 model that originally achieved FID 2.07 [9] to 1.55 that is very close to the state-of-the-art; previously, FID 1.48 has been reported for cascaded diffusion [16], 1.55 for classifier-free guidance [17], and 1.52 for StyleGAN-XL [38]. While our results showcase the potential gains achievable through sampler improvements, they also highlight the main shortcoming of stochasticity: For best results, one must make several heuristic choices — either implicit or explicit — that depend on the specific model. Indeed, we had to find the optimal values of  $\{S_{\text{churn}}, S_{\text{tmin}}, S_{\text{tmax}}, S_{\text{noise}}\}$  on a case-by-case basis using grid search (Appendix E). This raises a general concern that using stochastic sampling as the primary means of evaluating model improvements may inadvertently end up influencing the design choices related to model architecture and training.

# 5 Preconditioning and training

There are various known good practices for training neural networks in a supervised fashion. For example, it is advisable to keep input and output signal magnitudes fixed to, e.g., unit variance, and to avoid large variation in gradient magnitudes on a per-sample basis [5, 20]. Training a neural network to model  $D$  directly would be far from ideal—for example, as the input  $x = y + n$  is a combination

Table 2: Evaluation of our training improvements. The starting point (config A) is VP & VE using our deterministic sampler. At the end (configs E,F), VP & VE only differ in the architecture of  $F_{\theta}$ .  

<table><tr><td rowspan="2"></td><td colspan="4">CIFAR-10 [27] at 32×32</td><td rowspan="2" colspan="2">FFHQ [25] 64×64 Unconditional</td><td rowspan="2" colspan="2">AFHQv2 [7] 64×64 Unconditional</td></tr><tr><td colspan="2">Conditional</td><td colspan="2">Unconditional</td></tr><tr><td>Training configuration</td><td>VP</td><td>VE</td><td>VP</td><td>VE</td><td>VP</td><td>VE</td><td>VP</td><td>VE</td></tr><tr><td>A Baseline [41] (* pre-trained)</td><td>2.48</td><td>3.11</td><td>3.01*</td><td>3.77*</td><td>3.39</td><td>25.95</td><td>2.58</td><td>18.52</td></tr><tr><td>B + Adjust hyperparameters</td><td>2.18</td><td>2.48</td><td>2.51</td><td>2.94</td><td>3.13</td><td>22.53</td><td>2.43</td><td>23.12</td></tr><tr><td>C + Redistribute capacity</td><td>2.08</td><td>2.52</td><td>2.31</td><td>2.83</td><td>2.78</td><td>41.62</td><td>2.54</td><td>15.04</td></tr><tr><td>D + Our preconditioning</td><td>2.09</td><td>2.64</td><td>2.29</td><td>3.10</td><td>2.94</td><td>3.39</td><td>2.79</td><td>3.81</td></tr><tr><td>E + Our loss function</td><td>1.88</td><td>1.86</td><td>2.05</td><td>1.99</td><td>2.60</td><td>2.81</td><td>2.29</td><td>2.28</td></tr><tr><td>F + Non-leaky augmentation</td><td>1.79</td><td>1.79</td><td>1.97</td><td>1.98</td><td>2.39</td><td>2.53</td><td>1.96</td><td>2.16</td></tr><tr><td>NFE</td><td>35</td><td>35</td><td>35</td><td>35</td><td>79</td><td>79</td><td>79</td><td>79</td></tr></table>

of clean signal  $\mathbf{y}$  and noise  $\pmb{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ , its magnitude varies immensely depending on noise level  $\sigma$ . For this reason, the common practice is to not represent  $D_{\theta}$  as a neural network directly, but instead train a different network  $F_{\theta}$  from which  $D_{\theta}$  is derived.

Previous methods [32, 39, 41] address the input scaling via a  $\sigma$ -dependent normalization factor and attempt to precondition the output by training  $F_{\theta}$  to predict  $\pmb{n}$  scaled to unit variance, from which the signal is then reconstructed via  $D_{\theta}(\pmb{x};\sigma) = \pmb{x} - \sigma F_{\theta}(\cdot)$ . This has the drawback that at large  $\sigma$ , the network needs to fine-tune its output carefully to cancel out the existing noise  $\pmb{n}$  exactly and give the output at the correct scale; note that any errors made by the network are amplified by a factor of  $\sigma$ . In this situation, it would seem much easier to predict the expected output  $D(\pmb{x};\sigma)$  directly. To this end, we propose to precondition the neural network with a  $\sigma$ -dependent skip connection that allows it to estimate either  $\pmb{y}$  or  $\pmb{n}$ , or something in between. We thus write  $D_{\theta}$  in the following form:

$$
D _ {\theta} (\boldsymbol {x}; \sigma) = c _ {\text {s k i p}} (\sigma) \boldsymbol {x} + c _ {\text {o u t}} (\sigma) F _ {\theta} \left(c _ {\text {i n}} (\sigma) \boldsymbol {x}; c _ {\text {n o i s e}} (\sigma)\right), \tag {7}
$$

where  $F_{\theta}$  is the neural network to be trained,  $c_{\mathrm{skip}}(\sigma)$  modulates the skip connection,  $c_{\mathrm{in}}(\sigma)$  and  $c_{\mathrm{out}}(\sigma)$  scale the input and output magnitudes, and  $c_{\mathrm{noise}}(\sigma)$  maps noise level  $\sigma$  into a conditioning input for  $F_{\theta}$ . Taking a weighted expectation of Eq. 2 over the noise levels gives the overall training loss  $\mathbb{E}_{\sigma ,\boldsymbol {y},\boldsymbol{n}}\left[\lambda (\sigma)\| D(\boldsymbol {y} + \boldsymbol {n};\sigma) - \boldsymbol {y}\| _2^2\right]$ , where  $\sigma \sim p_{\mathrm{train}},\boldsymbol {y}\sim p_{\mathrm{data}}$ , and  $\pmb {n}\sim \mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$ . The probability of sampling a given noise level  $\sigma$  is given by  $p_{\mathrm{train}}(\sigma)$  and the corresponding weight is given by  $\lambda (\sigma)$ . We can equivalently express this loss with respect to the raw network output  $F_{\theta}$  in Eq. 7:

$$
\mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \left[ \underbrace {\lambda (\sigma) c _ {\text {o u t}} (\sigma) ^ {2}} _ {\text {e f f e c t i v e w e i g h t}} \| \underbrace {F _ {\theta} \left(c _ {\text {i n}} (\sigma) \cdot (\boldsymbol {y} + \boldsymbol {n}) ; c _ {\text {n o i s e}} (\sigma)\right)} _ {\text {n e t w o r k o u t p u t}} - \underbrace {\frac {1}{c _ {\text {o u t}} (\sigma)} \left(\boldsymbol {y} - c _ {\text {s k i p}} (\sigma) \cdot (\boldsymbol {y} + \boldsymbol {n})\right)} _ {\text {e f f e c t i v e t r a i n i n g t a r g e t}} \| _ {2} ^ {2} \right]. \tag {8}
$$

This form reveals the effective training target of  $F_{\theta}$ , allowing us to determine suitable choices for the preconditioning functions from first principles. As detailed in Appendix B, we derive our choices shown in Table 1 by requiring network inputs and training targets to have unit variance ( $c_{\mathrm{in}}$ ,  $c_{\mathrm{out}}$ ), and amplifying errors in  $F_{\theta}$  as little as possible ( $c_{\mathrm{skip}}$ ). The formula for  $c_{\mathrm{noise}}$  is chosen empirically.

Table 2 shows FID for a series of training setups, evaluated using our deterministic sampler from Section 3. We start with the baseline training setup of Song et al. [41], which differs considerably between the VP and VE cases; we provide separate results for each (config A). To obtain a more meaningful point of comparison, we re-adjust the basic hyperparameters (config B) and improve the expressive power of the model (config C) by removing the lowest-resolution layers and doubling the capacity of the highest-resolution layers instead; see Appendix F for further details. We then replace the original choices of  $\{c_{\mathrm{in}}, c_{\mathrm{out}}, c_{\mathrm{noise}}, c_{\mathrm{skip}}\}$  with our preconditioning (config D), which keeps the results largely unchanged—except for VE that improves considerably at  $64 \times 64$  resolution. Instead of improving FID per se, the main benefit of our preconditioning is that it makes the training more robust, enabling us to turn our focus on redesigning the loss function without adverse effects.

Loss weighting and sampling. Eq. 8 shows that training  $F_{\theta}$  as preconditioned in Eq. 7 incurs an effective per-sample loss weight of  $\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2$ . To balance the effective loss weights, we set  $\lambda(\sigma) = 1/c_{\mathrm{out}}(\sigma)^2$ , which also equalizes the initial training loss over the entire  $\sigma$  range as shown in Figure 5a (green curve). Finally, we need to select  $p_{\mathrm{train}}(\sigma)$ , i.e., how to choose noise levels during training. Inspecting the per-  $\sigma$  loss after training (blue and orange curves) reveals that a significant reduction is possible only at intermediate noise levels; at very low levels, it is both difficult and

![](images/8d2eb05efafb7d277cc59ccb412063616840c3bb9746e71c0eccf5c12681983c.jpg)  
(a) Training loss & noise distribution

![](images/d0ab036dc84b96fad3decc3eff696563a965c2e4b6a504239eb18391c4bd22b5.jpg)  
Figure 5: (a) Observed initial (green) and final loss per noise level, representative of the  $32 \times 32$  (blue) and  $64 \times 64$  (orange) models considered in this paper. The shaded regions represent the standard deviation over  $10\mathrm{k}$  random samples. Our proposed training sample density is shown by the dashed red curve. (b) For the original training setup of Song et al. [41], stochastic sampling is highly beneficial (blue, green), while deterministic sampling ( $S_{\mathrm{churn}} = 0$ ) leads to relatively poor FID. For our training setup, the situation is reversed (orange, red); stochastic sampling is not only unnecessary but harmful.  
(b) Amount of stochasticity on CIFAR-10

irrelevant to discern the vanishingly small noise component, whereas at high levels the training targets are always dissimilar from the correct answer that approaches dataset average. Therefore, we target the training efforts to the relevant range using a simple log-normal distribution for  $p_{\mathrm{train}}(\sigma)$  as detailed in Table 1 and illustrated in Figure 5a (red curve).

Table 2 shows that our proposed  $p_{train}$  and  $\lambda$  (config E) lead to a dramatic improvement in FID in all cases when used in conjunction with our preconditioning (config D). In concurrent work, Choi et al. [6] propose a similar scheme to prioritize noise levels that are most relevant w.r.t. forming the perceptually recognizable content of the image. However, they only consider the choice of  $\lambda$  in isolation, which results in a smaller overall improvement.

Augmentation regularization. To prevent potential overfitting that often plagues diffusion models with smaller datasets, we borrow an augmentation pipeline from the GAN literature [24]. The pipeline consists of various geometric transformations (see Appendix F) that we apply to a training image prior to adding noise. To prevent the augmentations from leaking to the generated images, we provide the augmentation parameters as a conditioning input to  $F_{\theta}$ ; during inference we set the them to zero to guarantee that only non-augmented images are generated. Table 2 shows that data augmentation provides a consistent improvement (config F) that yields new state-of-the-art FIDs of 1.79 and 1.97 for conditional and unconditional CIFAR-10, beating the previous records of 1.85 [38] and 2.10 [43].

# 6 Conclusions

Our approach of putting diffusion models to a common framework exposes a modular design. This allows a targeted investigation of individual components, potentially helping to better cover the viable design space. In our tests this let us to simply replace the samplers in various earlier models, drastically improving the results. For example, in ImageNet-64 our sampler turned an average model (FID 2.07) to a challenger (1.55) for the current SOTA model (1.48) [16]. We also obtained new state-of-the-art results on CIFAR-10 while using only 35 model evaluations, deterministic sampling, and a small network. The current high-resolution diffusion models rely either on separate super-resolution steps [16, 31, 35], subspace projection [22], very large networks [9, 41], or hybrid approaches [34, 36, 43]—we believe that our contributions are orthogonal to these extensions. That said, many of our parameter values may need to be re-adjusted for higher resolution datasets.

Interestingly, the relevance of stochastic sampling appears to diminish as the model itself improves, as shown in Figure 5b. Intuitively the role of stochasticity is to correct approximation errors, and if the approximation errors are minimal to begin with, there is nothing left to do. Nevertheless, we feel that the precise interaction between stochasticity and the training objective remains a promising avenue for future work, and stochasticity likely continues to be beneficial for more diverse datasets.

Negative societal impacts Our advances in sample quality can potentially amplify negative societal effects when used in a large-scale system like DALL-E 2, including types of disinformation or emphasizing stereotypes and harmful biases [29]. The training and sampling of diffusion models needs a lot of electricity; our project consumed  $\sim 250\mathrm{MWh}$  on an in-house cluster of NVIDIA V100s.

# References

[1] B. D. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313-326, 1982.  
[2] U. M. Ascher and L. R. Petzold. Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations. Society for Industrial and Applied Mathematics, 1998.  
[3] F. Bao, C. Li, J. Zhu, and B. Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In Proc. ICLR, 2022.  
[4] D. Baranchuk, A. Voynov, I. Rubachev, V. Khrulkov, and A. Babenko. Label-efficient semantic segmentation with diffusion models. In Proc. ICLR, 2022.  
[5] C. M. Bishop. Neural networks for pattern recognition. Oxford University Press, USA, 1995.  
[6] J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon. Perception prioritized training of diffusion models. CoRR, abs/2204.00227, 2022.  
[7] Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha. StarGAN v2: Diverse image synthesis for multiple domains. In Proc. CVPR, 2020.  
[8] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proc. CVPR, pages 248–255. IEEE, 2009.  
[9] P. Dhariwal and A. Q. Nichol. Diffusion models beat GANs on image synthesis. In Proc. NeurIPS, 2021.  
[10] J. R. Dormand and P. J. Prince. A family of embedded Runge-Kutta formulae. Journal of computational and applied mathematics, 6(1):19-26, 1980.  
[11] J. B. J. Fourier, G. Darboux, et al. Théorie analytique de la chaleur, volume 504. Didot Paris, 1822.  
[12] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks. In Proc. NIPS, 2014.  
[13] U. Grenander and M. I. Miller. Representations of knowledge in complex systems. Journal of the Royal Statistical Society: Series B (Methodological), 56(4):549-581, 1994.  
[14] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Proc. NIPS, 2017.  
[15] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Proc. NeurIPS, 2020.  
[16] J. Ho, C. Sahara, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23, 2022.  
[17] J. Ho and T. Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.  
[18] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet. Video diffusion models. CoRR, abs/2204.03458, 2022.  
[19] C.-W. Huang, J. H. Lim, and A. C. Courville. A variational perspective on diffusion-based generative models and score matching. In Proc. NeurIPS, 2021.  
[20] L. Huang, J. Qin, Y. Zhou, F. Zhu, L. Liu, and L. Shao. Normalization techniques in training DNNs: Methodology, analysis and application. CoRR, abs/2009.12836, 2020.  
[21] A. Hyvarinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695-709, 2005.  
[22] B. Jing, G. Corso, R. Berlinghieri, and T. Jaakkola. Subspace diffusion generative models. CoRR, abs/2205.01490, 2022.  
[23] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas. Gotta go fast when generating data with score-based models. CoRR, abs/2105.14080, 2021.  
[24] T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila. Training generative adversarial networks with limited data. In Proc. NeurIPS, 2020.  
[25] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In Proc. CVPR, 2018.  
[26] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In Proc. ICLR, 2021.  
[27] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.  
[28] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila. Noise2Noise: Learning image restoration without clean data. In Proc. ICML, 2018.  
[29] P. Mishkin, L. Ahmad, M. Brundage, G. Krueger, and G. Sastry. DALL-E 2 preview - risks and limitations. OpenAI, 2022.  
[30] E. Nachmani and S. Dovrat. Zero-shot translation using diffusion models. CoRR, abs/2111.01471, 2021.  
[31] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. CoRR, abs/2112.10741, 2021.  
[32] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In Proc. ICML, volume 139, pages 8162-8171, 2021.

[33] V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, and M. Kudinov. Grad-TTS: A diffusion probabilistic model for text-to-speech. In Proc. ICML, volume 139, pages 8599-8608, 2021.  
[34] K. Preechakul, N. Chathee, S. Wizadwongsa, and S. Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proc. CVPR, 2022.  
[35] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with CLIP latents. Technical report, OpenAI, 2022.  
[36] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proc. CVPR, 2022.  
[37] C. Sahara, W. Chan, H. Chang, C. A. Lee, J. Ho, T. Salimans, D. J. Fleet, and M. Norouzi. Palette: Image-to-image diffusion models. CoRR, abs/2111.05826, 2021.  
[38] A. Sauer, K. Schwarz, and A. Geiger. StyleGAN-XL: Scaling StyleGAN to large diverse datasets. CoRR, abs/2201.00273, 2022.  
[39] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In Proc. ICLR, 2021.  
[40] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. In Proc. NeurIPS, 2019.  
[41] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In Proc. ICLR, 2021.  
[42] E. Süli and D. Mayers. An Introduction to Numerical Analysis. Cambridge University Press, 2003.  
[43] A. Vahdat, K. Kreis, and J. Kautz. Score-based generative modeling in latent space. In Proc. NeurIPS, 2021.  
[44] P. Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661-1674, 2011.  
[45] D. Watson, W. Chan, J. Ho, and M. Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In Proc. ICLR, 2022.  
[46] D. Watson, J. Ho, M. Norouzi, and W. Chan. Learning to efficiently sample from diffusion probabilistic models. CoRR, abs/2106.03802, 2021.  
[47] J. Wolleb, R. Sandkuhler, F. Bieder, P. Valmaggia, and P. C. Cattin. Diffusion models for implicit image segmentation ensembles. In Medical Imaging with Deep Learning, 2022.  
[48] Q. Zhang and Y. Chen. Diffusion normalizing flow. In Proc. NeurIPS, 2021.
