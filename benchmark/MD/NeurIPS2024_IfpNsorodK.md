# PFDiff: Training-free Acceleration of Diffusion Models through the Gradient Guidance of Past and Future

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Diffusion Probabilistic Models (DPMs) have shown remarkable potential in image generation, but their sampling efficiency is hindered by the need for numerous denoising steps. Most existing solutions accelerate the sampling process by proposing fast ODE solvers. However, the inevitable discretization errors of the ODE solvers are significantly magnified when the number of function evaluations (NFE) is fewer. In this work, we propose PFDiff, a novel training-free and orthogonal timestep-skipping strategy, which enables existing fast ODE solvers to operate with fewer NFE. Based on two key observations: a significant similarity in the model's outputs at time step size that is not excessively large during the denoising process of existing ODE solvers, and a high resemblance between the denoising process and SGD. PFDiff, by employing gradient replacement from past time steps and foresight updates inspired by Nesterov momentum, rapidly updates intermediate states, thereby reducing unnecessary NFE while correcting for discretization errors inherent in first-order ODE solvers. Experimental results demonstrate that PFDiff exhibits flexible applicability across various pre-trained DPMs, particularly excelling in conditional DPMs and surpassing previous state-of-the-art training-free methods. For instance, using DDIM as a baseline, we achieved 16.46 FID (4 NFE) compared to 138.81 FID with DDIM on ImageNet 64x64 with classifier guidance, and 13.06 FID (10 NFE) on Stable Diffusion with 7.5 guidance scale.

# 1 Introduction

In recent years, Diffusion Probabilistic Models (DPMs) [1-4] have demonstrated exceptional modeling capabilities across various domains including image generation [5-7], video generation [8], text-to-image generation [9, 10], speech synthesis [11], and text-to-3D generation [12, 13]. They have become a key driving force advancing deep generative models. DPMs initiate with a forward process that introduces noise onto images, followed by utilizing a neural network to learn a backward process that incrementally removes noise, thereby generating images [2, 4]. Compared to other generative methods such as Generative Adversarial Networks (GANs) [14] and Variational Autoencoders (VAEs) [15], DPMs not only possess a simpler optimization target but also are capable of producing higher quality samples [5]. However, the generation of high-quality samples via DPMs requires hundreds or thousands of denoising steps, significantly lowering their sampling efficiency and becoming a major barrier to their widespread application.

Existing techniques for rapid sampling in DPMs primarily fall into two categories. First, training-based methods [16-19], which can significantly compress sampling steps, even achieving single-step sampling [19]. However, this compression often comes with a considerable additional training cost, and these methods are challenging to apply to large pre-trained models. Second, training-free samplers [20-30], which typically employ implicit or analytical solutions to Stochastic Differential Equations

Text Prompts: Winter night with snow -covered rooftops and soft yellow lights. (Left) A Corgi running towards me in Times Square. (Right)

![](images/ddae2e931dbc15968b7d79db1f226b3e921d0c46d9f18f9c18d12ac9b6cf2721.jpg)

![](images/22d565060b7e20cf66d13f7ec2f0de38ca03dafc33d66af2cba5a833445bfa96.jpg)  
(a) Results from Stable-Diffusion [9] on MS-COCO2014 [31] (Classifier-Free Guidance,  $s = 7.5$ )

![](images/ac6457658eb606f7f4a4c305df3270225462474ce49cd552b7429f44c9adc85d.jpg)  
(b) Results from Guided-Diffusion [5] on ImageNet 64x64 [32] (Classifier Guidance,  $s = 1.0$ )  
Figure 1: Sampling by conditional pre-trained DPMs [5, 9] using DDIM [20] and our method PFDiff (dashed box) with DDIM as a baseline, varying the number of function evaluations (NFE).

(SDE)/Ordinary Differential Equations (ODE) for lower-error sampling processes. For instance, Lu et al. [21, 22], by analyzing the semi-linear structure of the ODE solvers for DPMs, have sought to analytically derive optimally the solutions for DPMs' ODE solvers. These training-free sampling strategies can often be used in a plug-and-play fashion, compatible with existing pre-trained DPMs. However, when the NFE is below 10, the discretization error of these training-free methods will be significantly amplified, leading to convergence issues [21, 22], which can still be time-consuming.

To further enhance the sampling speed of DPMs, we have analyzed the potential for improvement in existing training-free accelerated methods. Initially, we observed a notably high similarity in the model's outputs for the existing ODE solvers of DPMs when time step size  $\Delta t$  is not extremely large, as illustrated in Fig. 2a. This observation led us to utilize the gradients that have been computed from past time steps to approximate current gradients, thereby reducing unnecessary estimation of noise network. Furthermore, due to the similarities between the sampling process of DPMs and Stochastic Gradient Descent (SGD) [33] as noted in Remark 1, we incorporated a foresight update mechanism using Nesterov momentum [34], known for accelerating SGD training. Specifically, we ingeniously employ prior observation to predict future gradients, then utilize the future gradients as a "springboard" to facilitate larger update step size  $\Delta t$ , as shown in Fig. 2b.

Motivated by these insights, we propose PFDiff, a timestep-skipping sampling algorithm that rapidly updates the current intermediate state through the gradient guidance of past and future. Notably, PFDiff is training-free and orthogonal to existing DPMs sampling algorithms, providing a new orthogonal axis for DPMs sampling. Unlike previous orthogonal sampling algorithms that compromise

![](images/dbe8a87eefcc47f159495401cde5f6d77203a8117df08de06625a5707b6cec59.jpg)  
(a) Gradient Changes in SDE/ODE Solvers

![](images/7067484cf4a08b01114e2e55a3807e1b7c1c3121ded210b604b6a0a9fc6ed7e0.jpg)  
Figure 2: (a) The trend of the MSE of the noise network output  $\epsilon_{\theta}(x_t,t)$  over time step size  $\Delta t$  where  $\eta$  in DDPM [2] comes from  $\bar{\sigma}_t$  in Eq. (6). Solid lines: ODE solvers, dashed lines: SDE solvers. (b) Comparison of partial sampling trajectories between PFDiff-1 and a first-order ODE solver, where the update directions are guided by the tangent direction of the sampling trajectories.  
(b) Comparison of Sampling Trajectories

sampling quality for speed [28], we prove that PFDiff corrects for errors in the sampling trajectories of first-order ODE solvers. This improves sampling quality while reducing unnecessary NFE in existing ODE solvers, as illustrated in Fig. 2b. To validate the orthogonality and effectiveness of PFDiff, extensive experiments were conducted on both unconditional [2, 4, 20] and conditional [5, 9] pre-trained DPMs, with the visualization experiment of conditional DPMs depicted in Fig. 1. The results indicate that PFDiff significantly enhances the sampling performance of existing ODE solvers. Particularly in conditional DPMs, PFDiff, using only DDIM as the baseline, surpasses the previous state-of-the-art training-free sampling algorithms.

# 2 Background

# 2.1 Diffusion SDEs

Diffusion Probabilistic Models (DPMs) [1-4] aim to generate  $D$ -dimensional random variables  $x_0 \in \mathbb{R}^D$  that follow a data distribution  $q(x_0)$ . Taking Denoising Diffusion Probabilistic Models (DDPM) [2] as an example, these models introduce noise to the data distribution through a forward process defined over discrete time steps, gradually transforming it into a standard Gaussian distribution  $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ . The forward process's latent variables  $\{x_t\}_{t \in [0,T]}$  are defined as follows:

$$
q \left(x _ {t} \mid x _ {0}\right) = \mathcal {N} \left(x _ {t} \mid \alpha_ {t} x _ {0}, \sigma_ {t} ^ {2} \boldsymbol {I}\right), \tag {1}
$$

where  $\alpha_{t}$  is a scalar function related to the time step  $t$ , with  $\alpha_{t}^{2} + \sigma_{t}^{2} = 1$ . In the model's reverse process, DDPM utilizes a neural network model  $p_{\theta}(x_{t-1} \mid x_t)$  to approximate the transition probability  $q(x_{t-1} \mid x_t, x_0)$ ,

$$
p _ {\theta} \left(x _ {t - 1} \mid x _ {t}\right) = \mathcal {N} \left(x _ {t - 1} \mid \mu_ {\theta} \left(x _ {t}, t\right), \sigma_ {\theta} ^ {2} (t) I\right), \tag {2}
$$

where  $\sigma_{\theta}^{2}(t)$  is defined as a scalar function related to the time step  $t$ . By sampling from a standard Gaussian distribution and utilizing the trained neural network, samples following the data distribution  $p_{\theta}(x_0) = \prod_{t=1}^{T} p_{\theta}(x_{t-1} \mid x_t)$  can be generated.

Furthermore, Song et al. [4] introduced SDE to model DPMs over continuous time steps, where the forward process is defined as:

$$
\mathrm {d} x _ {t} = f (t) x _ {t} \mathrm {d} t + g (t) \mathrm {d} w _ {t}, \quad x _ {0} \sim q \left(x _ {0}\right), \tag {3}
$$

where  $w_{t}$  represents a standard Wiener process, and  $f$  and  $g$  are scalar functions of the time step  $t$ . It's noteworthy that the forward process in Eq. (1) is a discrete form of Eq. (3), where  $f(t) = \frac{\mathrm{d}\log\alpha_t}{\mathrm{d}t}$  and  $g^{2}(t) = \frac{\mathrm{d}\sigma_{t}^{2}}{\mathrm{d}t} - 2\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t}\sigma_{t}^{2}$ . Song et al. [4] further demonstrated that there exists an equivalent reverse process from time step  $T$  to 0 for the forward process in Eq. (3):

$$
\mathrm {d} x _ {t} = \left[ f (t) x _ {t} - g ^ {2} (t) \nabla_ {x} \log q _ {t} \left(x _ {t}\right) \right] \mathrm {d} t + g (t) \mathrm {d} \bar {w} _ {t}, \quad x _ {T} \sim q \left(x _ {T}\right), \tag {4}
$$

where  $\bar{w}$  denotes a standard Wiener process. In this reverse process, the only unknown is the score function  $\nabla_{x}\log q_{t}(x_{t})$ , which can be approximated through neural networks.

# 2.2 Diffusion ODEs

In DPMs based on SDE, the discretization of the sampling process often requires a significant number of time steps to converge, such as the  $T = 1000$  time steps used in DDPM [2]. This requirement primarily stems from the randomness introduced at each time step by the SDE. To achieve a more efficient sampling process, Song et al. [4] utilized the Fokker-Planck equation [35] to derive a probability flow ODE related to the SDE, which possesses the same marginal distribution at any given time  $t$  as the SDE. Specifically, the reverse process ODE derived from Eq. (3) can be expressed as:

$$
\mathrm {d} x _ {t} = \left[ f (t) x _ {t} - \frac {1}{2} g ^ {2} (t) \nabla_ {x} \log q _ {t} \left(x _ {t}\right) \right] \mathrm {d} t, \quad x _ {T} \sim q \left(x _ {T}\right). \tag {5}
$$

Unlike SDE, ODE avoids the introduction of randomness, thereby allowing convergence to the data distribution in fewer time steps. Song et al. [4] employed a high-order RK45 ODE solver [36], achieving sample quality comparable to SDE at 1000 NFE with only 60 NFE. Furthermore, research such as DDIM [20] and DPM-Solver [21] explored discrete ODE forms capable of converging in fewer NFE. For DDIM, it breaks the Markov chain constraint on the basis of DDPM, deriving a new sampling formula expressed as follows:

$$
x _ {t - 1} = \sqrt {\alpha_ {t - 1}} \left(\frac {x _ {t} - \sqrt {1 - \alpha_ {t}} \epsilon_ {\theta} (x _ {t} , t)}{\sqrt {\alpha_ {t}}}\right) + \sqrt {1 - \alpha_ {t - 1} - \bar {\sigma} _ {t} ^ {2}} \epsilon_ {\theta} (x _ {t}, t) + \bar {\sigma} _ {t} \epsilon_ {t}, \tag {6}
$$

where  $\bar{\sigma}_t = \eta \sqrt{(1 - \alpha_{t-1}) / (1 - \alpha_t)}\sqrt{1 - \alpha_t / \alpha_{t-1}}$ , and  $\alpha_t$  corresponds to  $\alpha_t^2$  in Eq. (1). When  $\eta = 1$ , Eq. (6) becomes a form of DDPM [2]; when  $\eta = 0$ , it degenerates into an ODE, the form adopted by DDIM [20], which can obtain high-quality samples in fewer time steps.

Remark 1. In this paper, we regard the gradient  $\mathrm{d}\bar{x}_t$ , the noise network output  $\epsilon_{\theta}(x_t,t)$ , and the score function  $\nabla_x\log q_t(x_t)$  as expressing equivalent concepts. This is because Song et al. [4] demonstrated that  $\epsilon_{\theta}(x_t,t) = -\sigma_t\nabla_x\log q_t(x_t)$ . Moreover, we have discovered that any first-order solver of DPMs can be parameterized as  $x_{t-1} = \bar{x}_t - \gamma_t\mathrm{d}\bar{x}_t + \xi \epsilon_t$ . Taking DDIM [20] as an example, where  $\bar{x}_t = \sqrt{\frac{\alpha_{t-1}}{\alpha_t}} x_t$ ,  $\gamma_t = \sqrt{\frac{\alpha_{t-1}}{\alpha_t} - \alpha_{t-1}} - \sqrt{1 - \alpha_{t-1}}$ ,  $\mathrm{d}\bar{x}_t = \epsilon_{\theta}(x_t,t)$ , and  $\xi = 0$ . This indicates the similarity between SGD and the sampling process of DPMs, a discovery also implicitly suggested in the research of Xue et al. [30] and Wang et al. [37].

# 3 Method

# 3.1 Solving for reverse process diffusion ODEs

By substituting  $\epsilon_{\theta}(x_t,t) = -\sigma_t\nabla_x\log q_t(x_t)$  [4], Eq. (5) can be rewritten as:

$$
\frac {\mathrm {d} x _ {t}}{\mathrm {d} t} = s \left(\epsilon_ {\theta} \left(x _ {t}, t\right), x _ {t}, t\right) := f (t) x _ {t} + \frac {g ^ {2} (t)}{2 \sigma_ {t}} \epsilon_ {\theta} \left(x _ {t}, t\right), \quad x _ {T} \sim q \left(x _ {T}\right). \tag {7}
$$

Given an initial value  $x_{T}$ , we define the time steps  $\{t_i\}_{i=0}^T$  to progressively decrease from  $t_0 = T$  to  $t_T = 0$ . Let  $\tilde{x}_{t_0} = x_T$  be the initial value. Using  $T$  steps of iteration, we compute the sequence  $\{\tilde{x}_{t_i}\}_{i=0}^T$  to obtain the solution of this ODE. By integrating both sides of Eq. (7), we can obtain the exact solution of this sampling ODE.

$$
\tilde {x} _ {t _ {i}} = \tilde {x} _ {t _ {i - 1}} + \int_ {t _ {i - 1}} ^ {t _ {i}} s \left(\epsilon_ {\theta} \left(x _ {t}, t\right), x _ {t}, t\right) \mathrm {d} t. \tag {8}
$$

For any  $p$ -order ODE solver, Eq. (8) can be discretely represented as:

$$
\tilde {x} _ {t _ {i - 1} \rightarrow t _ {i}} \approx \tilde {x} _ {t _ {i - 1}} + \sum_ {n = 0} ^ {p - 1} h \left(\epsilon_ {\theta} \left(\tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right), \tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right) \cdot \Delta \hat {t}, \quad i \in [ 1, \dots , T ], \tag {9}
$$

where  $\hat{t}_0 = t_{i - 1},\hat{t}_p = t_i$  , and  $\Delta \hat{t} = \hat{t}_{n + 1} - \hat{t}_n$  denote the time step size. The function  $h$  represents the different solution methodologies applied by various  $p$  -order ODE solvers to the function  $s$  . For the Euler-Maruyama solver [38],  $h$  is the identity mapping of  $s$  . Further, we define

$\phi(Q, \tilde{x}_{t_{i-1}}, t_{i-1}, t_i) := \tilde{x}_{t_{i-1}} + \sum_{n=0}^{p-1} h(\epsilon_\theta(\tilde{x}_{\hat{t}_n}, \hat{t}_n), \tilde{x}_{\hat{t}_n}, \hat{t}_n) \cdot \Delta \hat{t}$ . Here,  $\phi$  is any  $p$ -order ODE solver, and buffer  $Q = \left(\{\epsilon_\theta(\tilde{x}_{\hat{t}_n}, \hat{t}_n)\}_{n=0}^{p-1}, t_{i-1}, t_i\right)$ , where  $\hat{t}_0 = t_{i-1}$  and  $\hat{t}_p = t_i$ .

When using the ODE solver defined in Eq. (9) for sampling, the choice of  $T = 1000$  leads to significant inefficiencies in DPMs. The study on DDIM [20] first revealed that by constructing a new forward sub-state sequence of length  $M + 1$  ( $M \leq T$ ),  $\{\tilde{x}_{t_i}\}_{i=0}^M$ , from a subsequence of time steps  $[0, \dots, T]$  and reversing this sub-state sequence, it is possible to converge to the data distribution in fewer time steps. However, as illustrated in Fig. 2a, for ODE solvers, as the time step  $\Delta t = t_i - t_{i-1}$  increases, the gradient direction changes slowly initially, but undergoes abrupt changes as  $\Delta t \to T$ . This phenomenon indicates that under minimal NFE (i.e., maximal time step size  $\Delta t$ ) conditions, the discretization error in Eq. (9) is significantly amplified. Consequently, existing ODE solvers, when sampling under minimal NFE, must sacrifice sampling quality to gain speed, making it an extremely challenging task to reduce NFE to below 10 [21, 22]. Given this, we aim to develop an efficient timestep-skipping sampling algorithm, which reduces NFE while correcting discretization errors, thereby ensuring that sampling quality is not compromised, and may even be improved.

# 3.2 Sampling guided by past gradients

For any  $p$ -order timestep-skipping sampling algorithm for DPMs, the sampling process can be reformulated according to Eq. (9) as follows:

$$
\tilde {x} _ {t _ {i}} \approx \phi (Q, \tilde {x} _ {t _ {i - 1}}, t _ {i - 1}, t _ {i}), \quad i \in [ 1, \dots , M ], \tag {10}
$$

where buffer  $Q = \left(\{\epsilon_{\theta}(\tilde{x}_{\hat{t}_n},\hat{t}_n)\}_{n = 0}^{p - 1},t_{i - 1},t_i\right)$  and  $[1,\dots ,M]$  is an increasing subsequence of  $[1,\ldots ,T]$ . As illustrated in Fig. 2a, when the time step size  $\Delta t$  (i.e.,  $t_i - t_{i - 1}$ ) is not excessively large, the MSE of the noise network, defined as  $\frac{1}{T - \Delta t}\sum_{t = 0}^{T - \Delta t - 1}\| \epsilon_{\theta}(x_t,t) - \epsilon_{\theta}(x_{t + \Delta t},t + \Delta t)\| ^2$ , is remarkably similar. This phenomenon is especially pronounced in ODE-based sampling algorithms, such as DDIM [20] and DPM-Solver [21]. This observation suggests that there are many unnecessary time steps in ODE-based sampling methods during the complete sampling process (e.g., when  $T = 1000$ ), which is one of the reasons these methods can generate samples in fewer steps. Based on this, we propose replacing the noise network of the current timestep with the output from a previous timestep to reduce unnecessary NFE without compromising the quality of the final generated samples. Initially, we store the output of the previous timestep's noise network in a buffer as follows:

$$
Q \xleftarrow {\text {b u f f e r}} \left(\left\{\epsilon_ {\theta} \left(\tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right) \right\} _ {n = 0} ^ {p - 1}, t _ {i - 1}, t _ {i}\right), \quad \text {w h e r e} \hat {t} _ {0} = t _ {i - 1}, \hat {t} _ {p} = t _ {i}. \tag {11}
$$

Then, in the current timestep, we directly use the noise network output saved in the buffer from the previous timestep to replace the current timestep's noise network output, thereby updating the intermediate states to the next timestep, as detailed below:

$$
\tilde {x} _ {t _ {i + 1}} \approx \phi (Q, \tilde {x} _ {t _ {i}}, t _ {i}, t _ {i + 1}), \quad \text {w h e r e} Q = \left(\left\{\epsilon_ {\theta} \left(\tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right) \right\} _ {n = 0} ^ {p - 1}, t _ {i - 1}, t _ {i}\right). \tag {12}
$$

By using this approach, we can effectively accelerate the sampling process, reduce unnecessary NFE, and ensure the quality of the samples is not affected. The convergence proof is in Appendix B.1.

# 3.3 Sampling guided by future gradients

As stated in Remark 1, considering the similarities between the sampling process of DPMs and SGD [33], we introduce a foresight update mechanism of Nesterov momentum, utilizing future gradient information as a "springboard" to assist the current intermediate state in achieving more efficient leapfrog updates. Specifically, for the intermediate state  $\tilde{x}_{t_{i+1}}$  predicted using past gradients as discussed in Sec. 3.2, we first estimate the future gradient and update the current buffer as follows:

$$
Q \xleftarrow {\text {b u f f e r}} \left(\left\{\epsilon_ {\theta} \left(\tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right) \right\} _ {n = 0} ^ {p - 1}, t _ {i + 1}, t _ {i + 2}\right), \quad \text {w h e r e} \hat {t} _ {0} = t _ {i + 1}, \hat {t} _ {p} = t _ {i + 2}. \tag {13}
$$

Subsequently, leveraging the concept of foresight updates, we predict a further future intermediate state  $\tilde{x}_{t_{i + 2}}$  using the current intermediate state  $\tilde{x}_{t_i}$  along with the future gradient information corresponding to  $\tilde{x}_{t_{i + 1}}$ , as shown below:

$$
\tilde {x} _ {t _ {i + 2}} \approx \phi (Q, \tilde {x} _ {t _ {i}}, t _ {i}, t _ {i + 2}), \quad \text {w h e r e} Q = \left(\left\{\epsilon_ {\theta} \left(\tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right) \right\} _ {n = 0} ^ {p - 1}, t _ {i + 1}, t _ {i + 2}\right). \tag {14}
$$

Furthermore, Zhou et al. [39] performed a Principal Component Analysis (PCA) on the sampling trajectories generated by ODE solvers for DPMs and discovered they almost lie in a two-dimensional plane embedded within a high-dimensional space. This implies that the Mean Value Theorem approximately holds during the sampling process using ODE solvers. Specifically, updating the current intermediate state  $\tilde{x}_{t_i}$  at an optimal time point  $s$  with the corresponding gradient information, ground truth  $\epsilon_{\theta}(\tilde{x}_{t_s}, t_s)$ , results in the smallest update error, where  $s$  is between time points  $i$  and  $i + 2$ . Further, we can reason that for any first-order ODE solver, under the same time step, the use of future gradient information  $\epsilon_{\theta}(\tilde{x}_{t_{i + 1}}, t_{i + 1})$  from Eq. (13) to update the current intermediate state  $\tilde{x}_{t_i}$  results in a smaller sampling error compared to using the gradient information at the current time point  $\epsilon_{\theta}(\tilde{x}_{t_i}, t_i)$ . A detailed proof is provided in Appendix B.2. However, for higher-order ODE solvers, the solving process implicitly utilizes future gradients as mentioned in Sec. 3.5, and the additional explicit introduction of future gradients increases sampling error. Therefore, when using higher-order ODE solvers as a baseline, the sampling process is accelerated by only using past gradients. It is only necessary to modify Eq. (14) to  $\tilde{x}_{t_{i + 2}} \approx \phi(Q, \tilde{x}_{t_{i + 1}}, t_{i + 1}, t_{i + 2})$  while keeping  $Q$  constant. Ablation experiments can be found in Sec. 4.3.

# 3.4 PFDiff: sampling guided by past and future gradients

Combining Sec. 3.2 and Sec. 3.3, the intermediate state  $\tilde{x}_{t_{i+1}}$  obtained through Eq. (12) is used to update the buffer  $Q$  in Eq. (13). In this way, we achieve our proposed efficient timestep-skipping algorithm, which we name PFDiff, as shown in Algorithm 1. For higher-order ODE solvers ( $p > 1$ ), PFDiff only utilizes past gradient information, while for first-order ODE solvers ( $p = 1$ ), it uses both past and future gradient information to predict further future intermediate states. Notably, during the iteration from intermediate state  $\tilde{x}_{t_i}$  to  $\tilde{x}_{t_{i+2}}$ , we only perform a single batch computation ( $NFE = p$ ) of the noise network in Eq. (13). Furthermore, we propose that in a single iteration process,  $\tilde{x}_{t_{i+2}}$  in Eq. (14) can be modified to  $\tilde{x}_{t_{i+(k+1)}}$ , achieving a  $k$ -step skip to sample more distant future intermediate states. Additionally, when  $k \neq 1$ , the buffer  $Q$ , which acts as an intermediate "springboard" from Eq. (13), has various computational origins. This can be accomplished by modifying  $\tilde{x}_{t_{i+1}}$  in Eq. (12) to  $\tilde{x}_{t_{i+l}}$ . We collectively refer to this multi-step skipping and different "springboard" selection strategy as PFDiff- $k\_l$  ( $l \leq k$ ). Further algorithmic details can be found in Appendix C. Finally, through the comparison of sampling trajectories between PFDiff-1 and a first-order ODE sampler, as shown in Fig. 2b, PFDiff-1 showcases its capability to correct the sampling trajectory of the first-order ODE sampler while reducing the NFE.

Proposition 3.1. For any given DPM first-order ODE solver  $\phi$ , the PFDiff- $k$ - $l$  algorithm can describe the sampling process within an iteration cycle through the following formula:

$$
\tilde {x} _ {t _ {i + (k + 1)}} \approx \phi \left(\epsilon_ {\theta} \left(\phi \left(\epsilon_ {\theta} \left(\tilde {x} _ {t _ {i - (k - l + 1)}}, t _ {i - (k - l + 1)}\right), \tilde {x} _ {t _ {i}}, t _ {i}, t _ {i + l}\right), t _ {i + l}\right), \tilde {x} _ {t _ {i}}, t _ {i}, t _ {i + (k + 1)}\right), \tag {15}
$$

# Algorithm 1 PFDiff-1

Require: initial value  $x_{T}$  , NFE  $N$  , model  $\epsilon_{\theta}$  , any  $p$  -order solver  $\phi$    
1: Define time steps  $\{t_i\}_{i = 0}^M$  with  $M = 2N - 1p$    
2:  $\tilde{x}_{t_0}\gets x_T$    
3:  $Q\stackrel {\mathrm{buffer}}{\longleftarrow}\left(\{\epsilon_{\theta}(\tilde{x}_{\hat{t}_n},\hat{t}_n)\}_{n = 0}^{p - 1},t_0,t_1\right),$  where  $\hat{t}_0 = t_0,\hat{t}_p = t_1$  ▷ Initialize buffer   
4:  $\tilde{x}_{t_1} = \phi (Q,\tilde{x}_{t_0},t_0,t_1)$    
5: for  $i\gets 1$  to  $\frac{M}{p} -2$  do   
6: if  $(i - 1)$  mod  $2 = 0$  then   
7:  $\tilde{x}_{t_{i + 1}} = \phi (Q,\tilde{x}_{t_i},t_i,t_{i + 1})$  ▷ Updating guided by past gradients   
8:  $Q\stackrel {\mathrm{buffer}}{\longleftarrow}\left(\{\epsilon_{\theta}(\tilde{x}_{\hat{t}_n},\hat{t}_n)\}_{n = 0}^{p - 1},t_{i + 1},t_{i + 2}\right)$  ▷ Update buffer (overwrite)   
9: if  $p = 1$  then   
10:  $\tilde{x}_{t_{i + 2}} = \phi (Q,\tilde{x}_{t_i},t_i,t_{i + 2})$  ▷ Anticipatory updating guided by future gradients   
11: else if  $p > 1$  then   
12:  $\tilde{x}_{t_{i + 2}} = \phi (Q,\tilde{x}_{t_{i + 1}},t_{i + 1},t_{i + 2})$  ▷ The higher-order solver uses only past gradients   
13: end if   
14: end if   
15: end for   
16: return  $\tilde{x}_{t_M}$

where the value of  $\epsilon_{\theta}(\tilde{x}_{t_{i - (k - l + 1)}},t_{i - (k - l + 1)})$  can be directly obtained from the buffer  $Q$ , without the need for additional computations. The iterative process defined by Eq. (15) ensures that the sampling outcomes converge to the data distribution consistent with the solver  $\phi$ , while effectively correcting errors in the sampling process (Proof in Appendix B).

It is noteworthy that, although the PFDiff is conceptually orthogonal to the SDE/ODE solvers of DPMs, even when the time size  $\Delta t$  is relatively small, the MSE of the noise network in the SDE solver exhibits significant differences, as shown in Fig. 2a. Consequently, PFDiff shows marked improvements on the ODE solver, and our experiments are almost exclusively based on ODE solvers, with exploratory experiments on SDE solvers referred to Sec. 4.1.

# 3.5 Connection with other samplers

Relationship with  $p$ -order solver [21, 22, 27]. According to Eq. (10), a single iteration of the  $p$ -order solver can be represented as:

$$
\tilde {x} _ {t _ {i + 1}} \approx \operatorname {S o l v e r} - \mathrm {p} \left(\left\{\epsilon_ {\theta} \left(\tilde {x} _ {\hat {t} _ {n}}, \hat {t} _ {n}\right) \right\} _ {n = 0} ^ {p - 1}, t _ {i}, t _ {i + 1}\right), \tilde {x} _ {t _ {i}}, t _ {i}, t _ {i + 1}), \quad i \in [ 0, \dots , M - 1 ]. \tag {16}
$$

A single iteration of the  $p$ -order solver uses  $p$  NFE to predict the next intermediate state. The intermediate step gradients obtained during this process can be considered as an approximation of future gradients. This approximation is implicitly contained within the sampling guided by future gradients that we propose. Furthermore, as shown in Eq. (15), a single iteration update of PFDiff based on a first-order solver can be seen as using a 2-order solver with only one NFE.

# 4 Experiments

In this section, we validate the effectiveness of PFDiff as an orthogonal and training-free sampler through a series of extensive experiments. This sampler can be integrated with any order of ODE solvers, thereby significantly enhancing the sampling efficiency of various types of pre-trained DPMs. To systematically showcase the performance of PFDiff, we categorize the pre-trained DPMs into two main types: conditional and unconditional. Unconditional DPMs are further subdivided into discrete and continuous, while conditional DPMs are subdivided into classifier guidance and classifier-free guidance. In choosing ODE solvers, we utilized the widely recognized first-order DDIM [20], Analytic-DDIM [23], and the higher-order DPM-Solver [21] as baselines. For each experiment, we use the Fréchet Inception Distance (FID↓) [40] as the primary evaluation metric, and provide the experimental results of the Inception Score (IS↑) [41] in the Appendix D.7 for reference. Lastly, apart from the ablation studies on parameters  $k$  and  $l$  discussed in Sec. 4.3, we showcase the optimal results of PFDiff- $k\_l$  (where  $k = 1, 2, 3$  and  $l \leq k$ ) across six configurations as a performance demonstration of PFDiff. As described in Appendix C, this does not increase the computational burden in practical applications. All experiments were conducted on an NVIDIA RTX 3090 GPU.

# 4.1 Unconditional sampling

For unconditional DPMs, we selected discrete DDPM [2] and DDIM [20], as well as pre-trained models from continuous ScoreSDE [4], to assess the effectiveness of PFDiff. For these pre-trained models, all experiments sampled 50k instances to compute evaluation metrics.

For unconditional discrete DPMs, we first select first-order ODE solvers DDIM [20] and Analytic-DDIM [23] as baselines, while implementing SDE-based DDPM [2] and Analytic-DDPM [23] methods for comparison, where  $\eta = 1.0$  is from  $\bar{\sigma}_t$  in Eq. (6). We conduct experiments on the CIFAR10 [42] and CelebA 64x64 [43] datasets using the quadratic time steps employed by DDIM. By varying the NFE from 6 to 20, the evaluation metric FID↓ is shown in Figs. 3a and 3b. Additionally, experiments with uniform time steps are conducted on the CelebA 64x64, LSUN-bedroom 256x256 [44], and LSUN-church 256x256 [44] datasets, with more results available in Appendix D.2. Our experimental results demonstrate that PFDiff, based on pre-trained models of discrete unconditional DPMs, significantly improves the sampling efficiency of DDIM and Analytic-DDIM samplers across multiple datasets. For instance, on the CIFAR10 dataset, PFDiff combined with DDIM achieves a FID of 4.10 with only 15 NFE, comparable to DDIM's performance of 4.04 FID with 1000 NFE. This is something other time-step skipping algorithms [23, 28] that sacrifice sampling quality for speed

![](images/f5a40bc01dff387adfbd66e46eb4f554c013d593316b0f66b5090acfc5b519e3.jpg)  
(a) CIFAR10 (Discrete)

![](images/4c24dc74f46d615c3721af541e6bc7a2a84bb51ec891b24f296ca4aa0653c556.jpg)  
(b) CelebA 64x64 (Discrete)

![](images/05cbb124fc6d66e4c84262f1a6f8845d32cc141b0fd1f966c751a286eda094d3.jpg)  
(c) CIFAR10 (Continuous)

![](images/4fa867c14ea95973ca06b1d0859ddd4cfe69b3b97048dce90c2f7a4eb24b9c6a.jpg)  
Figure 3: Unconditional sampling results. We report the FID↓ for different methods by varying the number of function evaluations (NFE), evaluated on 50k samples.  
(a) ImageNet 64x64 (Guided-Diffusion)  
(Classifier Guidance,  $s = 1.0$ )  
Figure 4: Conditional sampling results. We report the FID↓ for different methods by varying the NFE. Evaluated: ImageNet 64x64 with 50k, others with 10k samples. *AutoDiffusion [26] method requires additional search costs. †We borrow the results reported in DPM-Solver-v3 [27] directly.  
(Classifier-Free Guidance,  $s = 1.5$ )

![](images/92de990a61099cf2eaad9ae2b23caae220a358f1e450f5b81cbab8fad6473f7c.jpg)  
(b) MS-COCO2014 (Stable-Diffusion)

![](images/ac5eb4b01b22f010ab0d48190eaf132e93d4b1559871b1000cea01898d0393e5.jpg)  
(Classifier-Free Guidance,  $s = 7.5$ )  
(c) MS-COCO2014 (Stable-Diffusion)

cannot achieve. Furthermore, in Appendix D.2, by varying  $\eta$  from 1.0 to 0.0 in Eq. (6) to control the scale of noise introduced by SDE, we observe that as  $\eta$  decreases (reducing noise introduction), the performance of PFDiff gradually improves. This once again validates our assumption proposed in Sec. 3.2, based on Fig. 2a, that there is a significant similarity in the model's outputs at the time step size that is not excessively large for the existing ODE solvers.

For unconditional continuous DPMs, we choose the DPM-Solver-1, -2 and -3 [21] as the baseline to verify the effectiveness of PFDiff as an orthogonal timestep-skipping algorithm on the first and higher-order ODE solvers. We conducted experiments on the CIFAR10 [42] using quadratic time steps, varying the NFE. The experimental results using FID↓ as the evaluation metric are shown in Fig. 3c. More experimental details can be found in Appendix D.3. We observe that PFDiff consistently improves the sampling performance over the baseline with fewer NFE settings, particularly in cases where higher-order ODE solvers fail to converge with a small NFE (below 10) [21].

# 4.2 Conditional sampling

For conditional DPMs, we selected the pre-trained models of the widely recognized classifier guidance paradigm, ADM-G [5], and the classifier-free guidance paradigm, Stable-Diffusion [9], to validate the effectiveness of PFDiff. We employed uniform time steps setting and the DDIM [20] ODE solver as a baseline across all datasets. Evaluation metrics were computed by sampling 50k samples on the ImageNet 64x64 [32] dataset for ADM-G and 10k samples on other datasets, including ImageNet 256x256 [32] in ADM-G and MS-COCO2014 [31] in Stable-Diffusion.

For conditional DPMs employing the classifier guidance paradigm, we conducted experiments on the ImageNet 64x64 dataset [32] with a guidance scale  $(s)$  set to 1.0. For comparison, we implemented DPM-Solver-2 and -3 [21], and DPM-Solver++(2M) [22], which exhibit the best performance on conditional DPMs. Additionally, we introduced the AutoDiffusion method [26] using DDIM as a

baseline for comparison, noting that this method incurs additional search costs. We compared FID↓ scores by varying the NFE as depicted in Fig. 4a, with corresponding visual comparisons shown in Fig. 1b. We observed that PFDiff reduced the FID from 138.81 with 4 NFE in DDIM to 16.46, achieving an  $88.14\%$  improvement in quality. The visual results in Fig. 1b further demonstrate that, at the same NFE setting, PFDiff achieves higher-quality sampling. Furthermore, we evaluated PFDiff's sampling performance based on DDIM on the large-scale ImageNet 256x256 dataset [32]. Detailed results are provided in Appendix D.4.

For conditional, classifier-free guidance paradigms of DPMs, we employed the sd-v1-4 checkpoint and computed the FID↓ scores on the validation set of MS-COCO2014 [31]. We conducted experiments with a guidance scale (s) set to 7.5 and 1.5. For comparison, we implemented DPM-Solver-2 and -3 [21], and DPM-Solver++(2M) [22] methods. At s = 7.5, we introduced the state-of-the-art method reported in DPM-Solver-v3 [27] for comparison, along with DPM-Solver++(2M) [22], UniPC [29], and DPM-Solver-v3(2M) [27]. The FID↓ metrics by varying the NFE are presented in Figs. 4b and 4c, with additional visual results illustrated in Fig. 1a. We observed that PFDiff, solely based on DDIM, achieved state-of-the-art results during the sampling process of Stable-Diffusion, thus demonstrating the efficacy of PFDiff. Further experimental details can be found in Appendix D.5.

# 4.3 Ablation study

We conducted ablation experiments on the six different algorithm configurations of PFDiff mentioned in Appendix C, with  $k = 1,2,3$  ( $l \leq k$ ). Specifically, we evaluated the FID↓ scores on the unconditional and conditional pre-trained DPMs [2, 4, 5, 9] by varying the NFE. Detailed experimental setups and results can be found in Appendix D.6.1. The experimental results indicate that for various pre-trained DPMs, the choice of parameters  $k$  and  $l$  is not critical, as most combinations of  $k$  and  $l$  within PFDiff can enhance the sampling efficiency over the baseline. Moreover, with  $k = 1$  fixed, PFDiff-1 can significantly improve the baseline's sampling quality within the range of  $8 \sim 20$  NFE. For even better sampling quality, one can sample a small subset of examples (e.g., 5k) to compute evaluation metrics or directly conduct visual analysis, easily identifying the most effective  $k$  and  $l$  combinations.

To validate the PFDiff algorithm as mentioned in Sec. 3.3, which necessitates the joint guidance of past and future gradients for first-order ODE solvers, and only past gradients for higher-order ODE solvers, offering a more effective means of accelerating baseline sampling. This study employs the first-order ODE solver DDIM [20] as the baseline, isolating the effects of both past and future gradients, and uses the higher-order ODE solver DPM-Solver [21] as the baseline, removing the influence of future gradients for ablation experiments. Specific experimental configurations and results are shown in Appendix D.6.2. The results indicate that, as described by the PFDiff algorithm in Sec. 3.3, it is possible to further enhance the sampling efficiency of ODE solvers of any order.

# 5 Conclusion

In this paper, based on the recognition that the ODE solvers of DPMs exhibit significant similarity in model outputs when the time step size is not excessively large, and with the aid of a foresight update mechanism, we propose PFDiff, a novel method that leverages the gradient guidance from both past and future to rapidly update the current intermediate state. This approach effectively reduces the unnecessary number of function evaluations (NFE) in the ODE solvers and significantly corrects the errors of first-order ODE solvers during the sampling process. Extensive experiments demonstrate the orthogonality and efficacy of PFDiff on both unconditional and conditional pre-trained DPMs, especially in conditional pre-trained DPMs where PFDiff outperforms previous state-of-the-art training-free sampling methods.

Limitations and broader impact Although PFDiff can effectively accelerate the sampling speed of existing ODE solvers, it still lags behind the sampling speed of training-based acceleration methods and one-step generation paradigms such as GANs. Moreover, there is no universal setting for the optimal combination of parameters  $k$  and  $l$  in PFDiff; adjustments are required according to different pre-trained DPMs and NFE. It is noteworthy that PFDiff may be utilized to accelerate the generation of malicious content, thereby having a detrimental impact on society.

# References

[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256-2265. PMLR, 2015.  
[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840-6851, 2020.  
[3] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.  
[4] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.  
[5] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780-8794, 2021.  
[6] Jonathan Ho, Chitwan Sahara, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1-33, 2022.  
[7] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195-4205, 2023.  
[8] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n'pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024.  
[9] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684-10695, 2022.  
[10] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn.openai.com/papers/dall-e-3.pdf, 2(3):8, 2023.  
[11] Kaitao Song, Yichong Leng, Xu Tan, Yicheng Zou, Tao Qin, and Dongsheng Li. Transcormer: Transformer for sentence scoring with sliding language modeling. Advances in Neural Information Processing Systems, 35:11160-11174, 2022.  
[12] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.  
[13] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023.  
[14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.  
[15] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
[16] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.  
[17] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.  
[18] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. arXiv preprint arXiv:2206.02262, 2022.  
[19] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

[20] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.  
[21] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775-5787, 2022.  
[22] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.  
[23] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. arXiv preprint arXiv:2201.06503, 2022.  
[24] Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, and Bo Zhang. Estimating the optimal covariance with imperfect mean in diffusion probabilistic models. arXiv preprint arXiv:2206.07309, 2022.  
[25] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. arXiv preprint arXiv:2202.09778, 2022.  
[26] Lijiang Li, Huixia Li, Xiawu Zheng, Jie Wu, Xuefeng Xiao, Rui Wang, Min Zheng, Xin Pan, Fei Chao, and Rongrong Ji. Autodiffusion: Training-free optimization of time steps and architectures for automated diffusion model acceleration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7105-7114, 2023.  
[27] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. arXiv preprint arXiv:2310.13268, 2023.  
[28] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. arXiv preprint arXiv:2312.00858, 2023.  
[29] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.  
[30] Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhi-Ming Ma. Sa-solver: Stochastic adams solver for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.  
[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740-755. Springer, 2014.  
[32] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248-255. Ieee, 2009.  
[33] Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pages 400-407, 1951.  
[34] Yurii Nesterov. A method of solving a convex programming problem with convergence rate o (1/k** 2). Doklady Akademii Nauk SSSR, 269(3):543, 1983.  
[35] Bernt Øksendal and Bernt Øksendal. Stochastic differential equations. Springer, 2003.  
[36] John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. Journal of computational and applied mathematics, 6(1):19-26, 1980.  
[37] Kai Wang, Zhaopan Xu, Yukun Zhou, Zelin Zang, Trevor Darrell, Zhuang Liu, and Yang You. Neural network diffusion. arXiv preprint arXiv:2402.13144, 2024.  
[38] Peter E Kloeden, Eckhard Platen, Peter E Kloeden, and Eckhard Platen. Stochastic differential equations. Springer, 1992.  
[39] Zhenyu Zhou, Defang Chen, Can Wang, and Chun Chen. Fast ode-based sampling for diffusion models in around 5 steps. arXiv preprint arXiv:2312.00094, 2023.  
[40] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[41] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.  
[42] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
[43] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730-3738, 2015.  
[44] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.  
[45] Elliott Ward Cheney, EW Cheney, and W Cheney. Analysis for applied mathematics, volume 1. Springer, 2001.
