# Reliable Estimation of KL Divergence using a Discriminator in Reproducing Kernel Hilbert Space

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Estimating Kullback-Leibler (KL) divergence from samples of two distributions is essential in many machine learning problems. Variational methods using neural network discriminator have been proposed to achieve this task in a scalable manner. However, we noted that most of these methods using neural network discriminators suffer from high fluctuations (variance) in estimates and instability in training. In this paper, we look at this issue from statistical learning theory and function space complexity perspective to understand why this happens and how to solve it. We argue that the cause of these pathologies is lack of control over the complexity of the neural network discriminator function and could be mitigated by controlling it. To achieve this objective, we 1) present a novel construction of the discriminator in the Reproducing Kernel Hilbert Space (RKHS), 2) theoretically relate the error probability bound of the KL estimates to the complexity of the discriminator in the RKHS space, 3) present a scalable way to control the complexity (RKHS norm) of the discriminator for a reliable estimation of KL divergence, and 4) prove the consistency of the proposed estimator. In three different applications of KL divergence - estimation of KL, estimation of mutual information and Variational Bayes - we show that by controlling the complexity as developed in the theory, we are able to reduce the variance of KL estimates and stabilize the training.

# 1 Introduction

Estimating Kullback-Leibler (KL) divergence from data samples is an essential component in many machine learning problems including Bayesian inference, calculation of mutual information or methods using information theoretic objectives. Variational formulation of Bayesian Inference requires KL divergence computation, which could be challenging when we only have finite samples from two distributions. Similarly, computation of information theoretic objectives like mutual information requires computation of KL divergence between the joint and the product of marginals.

KL divergence estimation from samples was studied thoroughly by Nguyen et al. [1] using a variational technique, convex optimization and RKHS norm regularization, while also providing theoretical guarantees and insights. However, their technique requires handling the whole dataset at once and is not scalable. Many modern models need to use KL divergence with large scale data, and often with neural networks, for example total correlation variational autoencoder (TC-VAE) [2], adversarial variational Bayes (AVB) [3], information maximizing GAN (InfoGAN) [4], and amortized MAP [5] all need to compute KL divergence in a deep learning setup. These large scale models have imposed new requirements on KL divergence estimation like scalability (able to handle large amount of data samples) and minibatch compatibility (compatible with minibatch-based optimization).

Methods like Nguyen et al. [1] are not suitable in the large scale setup. These modern needs were later met by modern neural network based methods such as variational divergence minimization

(VDM) [6], mutual information neural estimation (MINE) [7], and discriminator based KL estimation with GAN-type objective [8, 5]. A key attribute of these methods is that they are based on updating a neural-net based discriminator to estimate KL divergence from a subset of samples making them scalable and minibatch compatible. We, however, noticed that even in simple examples, these methods exhibited pathologies like unreliability (high fluctuation of estimates) or instability during training (KL estimates blowing up). Similar observations of instability of VDM and MINE have also been reported in the literature [8, 9].

Why are these techniques unreliable? In this paper, we attempt to understand the core problem in the KL estimation using discriminator network. We look at it from the perspective of statistical learning theory and discriminator function space complexity and draw insights. Based on these insights, we propose that these fluctuations are a consequence of not controlling the smoothness and the complexity of the discriminator function space. Measuring and controlling the complexity of function space itself becomes a difficult problem when the discriminator is a deep neural network. Note that naive approaches to bound complexity by the number of parameters would neither be guaranteed to yield meaningful bound [10], nor be easy to implement.

Therefore, we present the following contributions to resolve these challenges. First, we propose a novel construction of the discriminator function using deep network such that it lies in a smooth function space, the Reproducing Kernel Hilbert Space(RKHS). By utilizing the learning theory and the complexity analysis of the RKHS space, we bound the probability of the error of KL-divergence estimates in terms of the radius of RKHS ball and kernel complexity. Using this bound, we propose a scalable way to control the complexity by penalizing the RKHS norm. This additional regularization of the complexity is still linear,  $(O(m))$  in time complexity with the number of data samples. Then, we prove consistency of the proposed KL estimator using ideas from empirical process theory. Experimentally, we demonstrate that the proposed way of controlling complexity significantly improves KL divergence estimation and significantly reduce the variance. In mutual information estimation, our method is competitive with the state-of-the-art method and in Variational Bayesian application, our method stabilizes training of MNIST dataset leading to sharp reconstruction.

# 2 Related Work

Nguyen et al. [1] used variational method to estimate KL divergence from samples of two distribution using convex risk minimization (CRM). They used the RKHS norm as a way to both measure and penalize the complexity of the variational function. However, their work required handling all data at once and solving a convex optimization problem which has time complexity in the order of  $O(m^3)$  and space complexity in the order of  $O(m^2)$ . Ahuja [11] used similar convex formulation in RKHS space and found it difficult to scale. VDM reformulated the f-Divergence objective using Fenchel duality and used a neural network to represent the variational function [6]. Although close in concept to [1], it is scalable since it uses a separate discriminator network and adversarial optimization. It, however, did not control the complexity of the neural-net function, and faced issues with stability.

One area of modern application of KL-divergence estimation is in computing mutual information, which is useful in applications such as stabilizing GANs [7]. MINE [7] also optimized a lower bound to KL divergence (Donsker-Varadhan representation). Similar to VDM, MINE used a neural network as the dual variational function: it is thus scalable, but without complexity control and is unstable. Another use of KL divergence is scalable variational inference (VI) as shown in AVB [8]. VI requires KL divergence estimation between the posterior and the prior, which becomes nontrivial when a sample based scalable estimation is required. AVB solved it using GAN-type adversarial formulation and a neural network discriminator. Similarly, [5] used GAN-type adversarial formulation to obtain KL divergence in amortized inference.

Chen et al. [2] proposed TC-VAE to improve disentanglement by penalizing the KL divergence between the marginal latent distribution and the product of marginals in each dimension. The KL divergence was computed by a minibatch-based sampling strategy that gives a biased estimate. Our work is close to Song et al. [9] who investigated the high variance in existing mutual information estimators and found that clipping the discriminator output is helpful in reducing variance. In our work, we take a principled way to connect variance to the complexity of discriminator function space and constrain it by penalizing its RKHS norm instead. None of the existing works considered looking at the discriminator function space, connecting its complexity to the unreliable KL-divergence estimation, or mitigating the problem by controlling the complexity.

# 3 Reproducing Kernel Hilbert Space

Let  $\mathcal{H}$  be a Hilbert space of functions  $f:\mathcal{X}\to \mathbb{R}$  defined on non-empty space  $\mathcal{X}$ . It is a Reproducing Kernel Hilbert Space (RKHS) if the evaluation functional,  $\delta_x:\mathcal{H}\to \mathbb{R}$ ,  $\delta_x:f\mapsto f(x)$ , is linear continuous  $\forall x\in \mathcal{X}$ . Every RKHS,  $\mathcal{H}_K$ , is associated with a unique positive definite kernel,  $K:\mathcal{X}\times \mathcal{X}\rightarrow \mathbb{R}$ , called the reproducing kernel [12], such that it satisfies:

$$
1. \forall x \in \mathcal {X}, K (., x) \in \mathcal {H} _ {K} \quad 2. \forall x \in \tilde {\mathcal {X}}, \forall f \in \mathcal {H} _ {K}, \langle f, K (., x) \rangle_ {\mathcal {H} _ {K}} = f (x)
$$

RKHS is often studied using a specific integral operator. Let  $\mathcal{L}_2(d\rho)$  be a space of functions  $f:\mathcal{X}\to \mathbb{R}$  that are square integrable with respect to a Borel probability measure  $d\rho$  on  $\mathcal{X}$ , we define an integral operator  $\mathcal{L}_K:\mathcal{L}_2(d\rho)\to \mathcal{L}_2(d\rho)$  [13, 14]:  $(\mathcal{L}_Kf)(x) = \int_{\mathcal{X}}f(y)K(x,y)d\rho (y)$ . This operator will be important in constructing a function in RKHS and in computing sample complexity.

# 4 Problem Formulation and Contribution

GAN-type Objective for KL Estimation: Let  $p(x)$  and  $q(x)$  be two probability density functions in space  $\mathcal{X}$  and we want to estimate their KL divergence using finite samples from each distribution in a scalable and minibatch compatible manner. As shown in [8, 5], this can be achieved by using a discriminator function. First, a discriminator  $f: \mathcal{X} \to \mathbb{R}$  is trained with the objective:

$$
f ^ {*} = \underset {f} {\operatorname {a r g m a x}} \left[ E _ {p (x)} \log \sigma (f (x)) + E _ {q (x)} \log (1 - \sigma (f (x))) \right] \tag {1}
$$

where  $\sigma$  is the Sigmoid function given by  $\sigma(x) = \frac{e^x}{1 + e^x}$ . Then it can be shown [8, 5] that the KL divergence  $KL(p(x)||q(x))$  is given by:  $KL(p(x)||q(x)) = E_{p(x)}[f^*(x)]$

Sources of Error: Eq. (1) is ambiguous in the sense that it is silent about the discriminator function space over which the optimization is carried out. Typically, a neural network is used as the discriminator. This implies that we are considering the space of functions represented by the neural network of given architecture as the hypothesis space, over which the maximization occurs in eq. (1). Hence, we must rewrite eq. (1) as

$$
f _ {h} ^ {*} = \underset {f \in h} {\operatorname {a r g m a x}} \left[ E _ {p (x)} \log \sigma (f (x)) + E _ {q (x)} \log (1 - \sigma (f (x))) \right] \tag {2}
$$

where  $h$  is the discriminator function space. Furthermore, we also approximate integrals in eq. (2) with the Monte Carlo estimate using finite number of samples, say  $m$ , from the distribution  $p$  and  $q$ .

$$
f _ {h} ^ {m} = \underset {f \in h} {\operatorname {a r g m a x}} \left[ \frac {1}{m} \sum_ {x _ {i} \sim p (x _ {i})} \log \sigma \left(f \left(x _ {i}\right)\right) + \frac {1}{m} \sum_ {x _ {j} \sim q (x _ {j})} \log \left(1 - \sigma \left(f \left(x _ {j}\right)\right)\right) \right] \tag {3}
$$

Similarly, we write KL estimate obtained from, respectively, infinite and finite samples as:

$$
K L (f) = E _ {p (x)} [ f (x) ], \quad K L _ {m} (f) = \frac {1}{m} \sum_ {x _ {i} \sim p \left(x _ {i}\right)} [ f (x) ] \tag {4}
$$

Each of these steps introduces some error in our estimate. We can now start our analysis by first decomposing the total estimation error as:

This equation decomposes total estimation error into three terms: 1) deviation from the mean error, 2) error in KL estimate by the discriminator due to using finite samples in optimization eq. (3), and 3) bias when the considered function space does not contain the optimal function. Here, we concentrate on quantifying the probability of deviation-from-mean error which is directly related to observed variance of the KL estimate.  
Summary of Technical Contributions: Since the deviation is the difference between a random variable and its mean, we can bound the probability of this error using concentration inequality and the complexity of the function space of  $f_h^m$ . To use smooth function space, we propose to construct a function out of neural networks such that it lies on RKHS (Section 5). Then, we bound the probability of deviation-from-mean error through the covering number of the RKHS space (Section 6.1), then control complexity (Section 6.2) and prove consistency of the proposed estimator (Section 7).

# 5 Constructing  $f$  in RKHS

The following theorem due to [15] paves a way for us to construct a neural function in RKHS.

Theorem 1. [[15] Appendix A] A function  $f \in \mathcal{L}_2(d\rho)$  is in Reproducing Kernel Hilbert Space,  $\mathcal{H}_K$ , if and only if it can be expressed as

$$
\forall x \in \mathcal {X}, f (x) = \int_ {\mathcal {W}} g (w) \psi (x, w) d \tau (w), \tag {6}
$$

for a certain function  $g: \mathcal{W} \to \mathbb{R}$  such that  $||g||_{\mathcal{L}_2(d\tau)}^2 < \infty$ . The RKHS norm of  $f$  satisfies  $||f||_{\mathcal{H}_K}^2 \leq ||g||_{\mathcal{L}_2(d\tau)}^2$  and the kernel  $K$  is given by

$$
K (x, t) = \int_ {\mathcal {W}} \psi (x, w) \psi (t, w) d \tau (w) \tag {7}
$$

Theorem 1 not only gives us a condition when a square integrable function is guaranteed to lie in RKHS, it also provides us with a recipe to construct a function in RKHS. We use this theorem with the neural networks as  $\psi$  and  $g$ . We sample  $w \sim \mathcal{N}(0, \gamma \mathrm{I})$  and pass it through two neural networks,  $\psi$  and  $g$ , where  $\psi$  takes  $x$  and  $w$  as two arguments and  $g$  takes only  $w$  as an argument. More precisely, we consider  $\psi(x, w) = \phi_{\theta}(x)^T w$ . The kernel  $K$ , as defined in eq. (7), can be obtained as:

$$
K _ {\theta} \left(x ^ {*}, t ^ {*}\right) = \int_ {\mathcal {W}} \phi_ {\theta} \left(x ^ {*}\right) ^ {T} w w ^ {T} \phi_ {\theta} \left(t ^ {*}\right) d \tau (w) = \gamma \phi_ {\theta} \left(x ^ {*}\right) ^ {T} \phi_ {\theta} \left(t ^ {*}\right) \tag {8}
$$

where  $E_{w\sim \mathcal{N}(0,\gamma \mathrm{I})}[ww^T] = \gamma \mathrm{I}$ . We sometimes denote the kernel  $K$  by  $K_{\theta}$  to emphasize that it is a function of neural network parameters,  $\theta$ .

Traditionally, kernel  $K$  remains fixed and the norm of the function  $f$  determines the complexity of the function space. In our formulation, both the RKHS kernel and its norm with respect to the kernel change during training since the kernel depends on neural network parameters,  $\theta$ . Therefore, the challenge is to tease out how neural parameters,  $\theta$ , affect the deviation-from-mean error in eq. (5).

# 6 Error Analysis and Control

Assumptions: Before starting our analysis, we list assumptions upon which our theory is based.

A1. The input domains  $\mathcal{X}$  and  $\mathcal{W}$  are compact.  
A2. The functions  $\phi_{\theta}$  and  $g$  are Lipschitz continuous with Lipschitz constants  $L_{\phi}$  and  $L_{g}$  respectively.  
A3. Higher order derivatives  $D_x^\alpha K(x,t)$  up to some high order  $\tau = h / 2$  of kernel  $K$  exist.

Assumptions A1 is satisfied in our experiments since we consider a bounded set in  $\mathbb{R}^n$  and  $\mathbb{R}^D$  as our domains. Similarly, A2 is satisfied since we enforce Lipschitz continuity of  $\phi$  and  $g$  by using spectral normalization [16]. Assumption A3 is a bit subtle. By the definition of  $K$  in eq.(8), higher order derivative of  $K$  exists iff higher order derivative of  $\phi_{\theta}$  exists. This is readily satisfied by deep networks with smooth activation functions, and is true everywhere except at origin for ReLU activation. Using the boundedness of the input domain and Lipschitz continuity, we show the following:

Proposition 1. Under the assumptions A1, A2, we have  $\sup_{K_{\theta}} K_{\theta}(x,t) < \infty$  and  $\| g\|_{\mathcal{L}_2(d\tau)}^2 < \infty$ .

# 6.1 Bounding the Error Probability of KL Estimates

Bounding the probability of deviation-from-mean error (eq. (5)) is tricky since, in our case, the kernel is not fixed and we are also optimizing over them. We bound it in two steps: 1) we derive a bound for a fixed kernel, 2) we take supremum of this bound over all the kernels parameterized by  $\theta$ .

For a fixed kernel, we first bound the probability of deviation-from-mean error in terms of the covering number in Lemma 1. We then use an estimate of the covering number of RKHS due to [14] to relate the bound to kernel  $K_{\theta}$  in Theorem 2, identifying the role of neural networks in this error bound.

Lemma 1. Let  $f_{\mathcal{H}_K}^m$  be the optimal discriminator function in an RKHS  $\mathcal{H}_K$  which is  $M$ -bounded. Let  $KL_m(f_{\mathcal{H}_K}^m) = \frac{1}{m}\sum_i f_{\mathcal{H}_K}^m (x_i)$  and  $KL(f_{\mathcal{H}_K}^m) = E_{p(x)}[f_{\mathcal{H}_K}^m (x)]$  be the estimate of  $KL$  divergence

from  $m$  samples and that by using the true distribution  $p(x)$  respectively. Then the probability of error at some accuracy level,  $\epsilon$ , is lower-bounded as:

$$
P r o b. (| K L _ {m} (f _ {\mathcal {H} _ {K}} ^ {m}) - K L (f _ {\mathcal {H} _ {K}} ^ {m}) | \leq \epsilon) \geq 1 - 2 \mathcal {N} (\mathcal {H} _ {K}, \frac {\epsilon}{4 \sqrt {S _ {K}}}) \exp (- \frac {m \epsilon^ {2}}{4 M ^ {2}})
$$

where  $\mathcal{N}(\mathcal{H}_K,\eta)$  denotes the covering number of an RKHS space  $\mathcal{H}_K$  with disks of radius  $\eta$ , and  $S_K = \sup_{x,t} K(x,t)$  which we refer to as kernel complexity.

Proof Sketch. We cover RKHS with discs of radius  $\eta = \frac{\epsilon}{4\sqrt{S_K}}$ . Within this radius, the deviation does not change too much. So, we can bound deviation probability at the center of disc and apply union bound over all the discs. To bound deviation probability at the center, we apply Hoeffding's inequality and applying union bound simply leads to counting number of discs which is exactly the covering number. See supplementary materials for the full proof.

Lemma 1 bounds the probability of error in terms of the covering number of the RKHS space. Note that the radius of the disc is inversely related to  $S_K$  which indicates how complex the RKHS space defined by the kernel  $K_{\theta}$  is. Here  $K_{\theta}$  depends on the neural network parameters  $\theta$ . Therefore, we denote  $S_K$  as a function of  $\theta$  as  $S_K(\theta)$  and term it kernel complexity. Next, we use Lemma 2 due to [14] to obtain an error bound in estimating KL divergence with finite samples in Theorem 2.

Lemma 2 ([14]). Let  $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$  be a  $\mathcal{C}^\infty$  Mercer kernel and the inclusion  $I_K: \mathcal{H}_K \hookrightarrow \mathcal{C}(\mathcal{X})$  be the compact embedding defined by  $K$  to the Banach space  $\mathcal{C}(\mathcal{X})$ . Let  $B_R$  be the ball of radius  $R$  in RKHS  $\mathcal{H}_K$ . Then  $\forall \eta > 0, R > 0, h > n$ , we have

$$
\ln \mathcal {N} \left(I _ {K} \left(B _ {R}\right), \eta\right) \leq \left(\frac {R C _ {h}}{\eta}\right) ^ {\frac {2 n}{h}} \tag {9}
$$

where  $\mathcal{N}$  gives the covering number of the space  $I_{K}(B_{R})$  with discs of radius  $\eta$ , and  $n$  represents the dimension of the input space  $\mathcal{X}$ .  $C_h$  is given by  $C_h = C_s\sqrt{||\mathcal{L}_s||}$  where  $\mathcal{L}_s$  is a linear embedding from square integrable space  $\mathcal{L}_2(d\rho)$  to the Sobolev space  $H^{h/2}$  and  $C_s$  is a constant.

To prove Lemma 2 [14], the RKHS space is embedded in the Sobolev Space  $H^{h/2}$  using  $\mathcal{L}_s$  and then the covering number of the Sobolev space is used. Thus the norm of  $\mathcal{L}_s$  and the degree of Sobolev space,  $h/2$ , appears in the covering number of a ball in  $\mathcal{H}_K$ . In Theorem 2, we use Lemma 1 and 2 to bound the estimation error of KL divergence.

Theorem 2. Let  $KL(f_{\mathcal{H}}^{m})$  and  $KL_{m}(f_{\mathcal{H}}^{m})$  be the estimates of  $KL$  divergence obtained by using true distribution  $p(x)$  and  $m$  samples respectively as described in Lemma 1, then the probability of error in the estimation at the error level  $\epsilon$  is given by:

$$
P r o b. (| K L _ {m} (f _ {\mathcal {H}} ^ {m}) - K L (f _ {\mathcal {H}} ^ {m}) | \leq \epsilon) \geq 1 - 2 \exp \left[ \left(\frac {4 R C _ {p} \sqrt {S _ {p} | | \mathcal {L} _ {p} | |}}{\epsilon}\right) ^ {\frac {2 n}{h}} - \frac {m \epsilon^ {2}}{4 M ^ {2}} \right]
$$

where  $C_p \sqrt{S_p||\mathcal{L}_p||} = \sup_{K_\theta} C_s \sqrt{S_K(\theta)||\mathcal{L}_s||}$ , i.e.  $C_p, S_p, \mathcal{L}_p$  correspond to a kernel for which the bound is maximum.

Proof. We prove this in two steps: First we obtain an error bound for a fixed kernel space and apply supremum over all  $\theta$ . For any RKHS  $\mathcal{H}_{K_{\theta}}$ , with fixed kernel  $K_{\theta}$ , we have

$$
\operatorname {P r o b}. \left(\left| K L _ {m} \left(f _ {\mathcal {H} _ {K _ {\theta}}} ^ {m}\right) - K L \left(f _ {\mathcal {H} _ {K _ {\theta}}} ^ {m}\right) \right| \geq \epsilon\right) \leq 2 \exp \left[ \left(\frac {4 R C _ {s} \sqrt {S _ {K} (\theta) \left\| \mathscr {L} _ {s} \right\|}}{\epsilon}\right) ^ {\frac {2 n}{h}} - \frac {m \epsilon^ {2}}{4 M ^ {2}} \right] \tag {10}
$$

We prove this error bound as follows. Lemma 2 gives the covering number of an RKHS ball of radius  $R$ , which we apply to Lemma 1. We fix the radius of discs to  $\eta = \frac{\epsilon}{4\sqrt{S_K}}$  in Lemma 1 and substitute  $C_h = C_s\sqrt{||\mathcal{L}_s(\theta)||}$  to obtain eq.(10).

Since we are continuously changing  $\theta$  during training, the kernel also changes. Hence, to find the upper bound over all possible kernels, we take the supremum over all kernels.

$$
\begin{array}{l} \operatorname {P r o b}. \left(\left| K L _ {m} \left(f _ {\mathcal {H}} ^ {m}\right) - K L \left(f _ {\mathcal {H}} ^ {m}\right) \right| \geq \epsilon\right) \leq \sup  _ {K _ {\theta}} \operatorname {P r o b}. \left(\left| K L _ {m} \left(f _ {\mathcal {H} _ {K _ {\theta}}} ^ {m}\right) - K L \left(f _ {\mathcal {H} _ {K _ {\theta}}} ^ {m}\right) \right| \geq \epsilon\right) (11) \\ \leq 2 \exp \left[ \left(\frac {4 R C _ {p} \sqrt {S _ {p} \left| \right| \mathscr {L} _ {p} \left| \right|}}{\epsilon}\right) ^ {\frac {2 n}{h}} - \frac {m \epsilon^ {2}}{4 M ^ {2}} \right] (12) \\ \end{array}
$$

where  $S_{p} = S_{K}(\theta_{p})$  and  $\mathcal{L}_p = \mathcal{L}_K(\theta_p)$ , i.e.,  $S_{p}$  and  $\mathcal{L}_p$  correspond to kernel complexity and Sobolev operator norm corresponding to optimal kernel  $K_{\theta_p}$  that extremizes eq. (11). Theorem statement readily follows from eq. (12)

Theorem 2 shows that the error increases exponentially with the radius of the RKHS space,  $R$ , complexity of the kernel  $S_K(\theta_p)$ , and the norm of the Sobolev space embedding operator  $\| \mathcal{L}_p\|$ . The Sobolev embedding operator,  $\mathcal{L}_p$ , is a mapping from  $\mathcal{L}_2(d\rho)$  to the Sobolev space  $H^{h / 2}$ . It can be shown [14] that the operator norm can be bounded as  $\| \mathcal{L}_p\| \leq \rho (\mathcal{X})\sum_{|\alpha |\leq h / 2}\sup_{x,t\in \mathcal{X}}(D_x^\alpha K_{\theta_p}(x,t))^2$ , where  $\rho$  is the measure of the input space  $\mathcal{X}$ . Therefore, the norm  $\| \mathcal{L}_p\|$  directly measures smoothness of  $K_{\theta_p}$  in terms of norm of its derivative in addition to the supremum value of  $K$ , while  $S_K(\theta_p)$  only depends on the supremum value of  $K_{\theta_p}$ .

# 6.2 Complexity Control

From Theorem 2, we see that the error probability could be decreased by decreasing  $R$ ,  $||\mathcal{L}_p||$  and  $S_K(\theta_p)$ . Using argument similar to the proof of Proposition 1, we can show that the Lipschitz constraint on  $\phi_{\theta}$  also affects  $S_K$  and may affect  $||\mathcal{L}_p||$ . In our experiments, however, we fix the Lipschitz constraints during optimization and do not change  $S_K$  and  $||\mathcal{L}_p||$  dynamically. Here, we focus on the norm,  $R$  from Theorem 2. To obtain the optimal discriminator  $f_h^m$ , we optimize the following objective with an extra penalization of the upper bound, i.e.  $||g||$  on the RKHS norm of  $f$ :

$$
f _ {h} ^ {m} = \underset {f \in h} {\operatorname {a r g m a x}} \frac {1}{m} \sum_ {x _ {i} \sim p (x _ {i})} \log \sigma \left(f \left(x _ {i}\right)\right) + \frac {1}{m} \sum_ {x _ {j} \sim q \left(x _ {j}\right)} \log \left(1 - \sigma \left(f \left(x _ {j}\right)\right)\right) - \frac {\lambda_ {0}}{m} \| g \| _ {\mathcal {L} _ {2} (d \tau)} ^ {2} \tag {13}
$$

The regularization term prevents the radius of RKHS ball from growing, maintaining a low error probability. Optimization of eq. (13) w.r.t. neural network parameters  $\theta$  allows dynamic control of the complexity of the discriminator function on the fly in a scalable and efficient way. Note that, computation of  $||g||_{\mathcal{L}_2(d\tau)}$  requires randomly sampling  $w \sim \mathcal{N}(0,\gamma \mathbf{I})$  and passing through neural network  $g$  independent of the data  $x_i, x_j$ . Therefore, if the computational complexity of optimization is  $O(m)$ , it will remain the same after incorporating this additional term, i.e. regularization does not increase asymptotic time complexity which is linear with the number of samples,  $m$ .

# 7 Variance and Consistency of the Estimate

# 7.1 Variance Analysis

Theorem 2 gives an upper bound on the probability of error. Intuitively, the variance and probability of error behave similarly for many distributions, i.e. higher variance might indicate higher probability of error. Below we quantify this intuition for a Gaussian distributed estimate:

Theorem 3. Let  $X = KL_{m}(f_{\mathcal{H}}^{m})$  be the estimated KL divergence using m samples as described in Theorem 2. Assuming that  $X$  follows a Gaussian distribution  $X \sim \mathcal{N}(\mu, \sigma)$ , we can obtain an upper bound on this variance of the estimate as follows:

$$
\sigma \leq \frac {\epsilon}{e r f ^ {- 1} \left[ - 4 \exp \left[ \left(\frac {4 R C _ {p} \sqrt {S _ {p} | | L _ {p} | |}}{\epsilon}\right) ^ {\frac {2 n}{h}} - \frac {m \epsilon^ {2}}{4 M ^ {2}} \right] + 1 \right]} \tag {14}
$$

where  $\text{erf}$  is the Gauss error function and is a monotonic function.

Obviously, this relation applies only to Gaussian distributed estimate, a strong assumption. However, Theorem 3 is presented for illustrative purpose. It suggests that by decreasing  $R$ , the radius of the RKHS ball, the variance of the estimate could be decreased. Experimentally, we observe that the variance decreases as we penalize the RKHS norm more, consistent with the spirit of Theorem 3.

![](images/39085e63e4950a70bfc5ceb21293721b328c7bc525cf9a2c6b96e99e6d8c70fb.jpg)  
Figure 1: a) Top scatter plot compares KL divergence estimates between a method using Neural network discriminator without complexity control (red) and that using RKHS discriminator with complexity control (blue); b) In the bottom, we show the effect of varying the regularization parameter  $\lambda$  on bias and variance while using the RKHS discriminator with complexity control as in eq.(13).

# 7.2 Consistency of Estimates

Here we show that the regularized objective leads to a consistent estimation.

Theorem 4. Let  $f^{*}$  and  $f^{m}$  be optimal discriminators as described in eq. (1) and eq. (13) respectively, and the KL estimate is given by  $KL(f) = E_{p(x)}[f(x)]$ ,  $KL_{m}(f) = \frac{1}{m}\sum_{x_{i}\sim p(x_{i})}[f(x)]$ . Then, in the limiting case as  $m\to \infty$ ,  $|KL_{m}(f_{h}^{m}) - KL(f^{*})|\to 0$ .

Proof Sketch. The difference between the true KL divergence and the estimated KL divergence can be divided into three terms as shown in eq. (5). We assume that our function space is rich enough to contain the true solution, driving bias to zero. From Theorem 2, we see that in the limiting case of  $m \to 0$ , the deviation-from-mean error goes to 0. Therefore, the key step that remains to be shown is that the discriminator induced error (second term in eq.(5)) also goes to 0 as  $m \to \infty$ .

It can be shown if we can prove that the optimal discriminator in eq. (13) approaches the optimal discriminator in eq. (2). To prove this, we show that the argument being maximized by  $f_{h}^{m}$  approaches the argument being maximized by  $f_{h}^{*}$  in the limiting case. To show this, we need to show that the function space,  $\log \sigma f$ , is Glivenko Cantelli [17], which we prove in following steps:

1. We show that  $f$  is Lipschitz continuous by definition and due to Lipschitz continuity of  $\phi_{\theta}$ . Then we show that  $\log \sigma f$  is Lipschitz continuous if  $f$  is Lipschitz continuous.  
2. Then we show that for a class of functions with Lipschitz constant  $L$ , the metric entropy,  $\log N$ , can be obtained in terms of  $L$  and entropy number of the bounded input space,  $\mathcal{X}$ .  
3. Since the metric entropy does not grow with the number of samples  $m$ , we show that  $\frac{1}{m} \log N \to 0$  which lets us show that  $\log \sigma f$  belongs to Glivenko Cantelli class of functions by using Theorem 2.4.3 from [17]. See supplementary material for the complete proof.

# 8 Experimental Results

We present results on three applications of KL divergence estimation: 1. KL estimation between simple Guaussian distributions, 2. Mutual information estimation, 3. Variational Bayes. In our experiments, the RKHS discriminator is constructed with  $\psi$  and  $g$  networks as described in Section 5, where the network  $\psi$  is very close to a regular neural network. In two experiments, we compare our results with the models using regular neural net discriminator to ensure that the difference in performance between RKHS and regular neural network is not due to architectural difference.

![](images/8a2177910745b62c402889cdf619ebfaafd30a2877e21c8611cac0b4dcc2793e.jpg)  
Figure 2: Comparing our method with CPC [18], convex risk minimization(NWJ) [1] and SMILE [9] regarding mutual information estimation between two variables.

KL Estimation between Two Gaussians We assume that we have finite sets of samples from two distributions. We further assume that we are required to apply minibatch based optimization. We consider estimating KL divergence between two Gaussian distributions in 2D, where we know the analytical KL divergence between the two distributions as the ground truth. We consider three different pairs of distributions corresponding to true KL divergence values of 1.3, 13.8 and 38.29, respectively and use  $m = 5000$  samples from each distribution to estimate KL in the finite case. We repeat the estimation experiments with random initialization 30 times and report the mean, standard deviation, scatter and box plots.

Fig. 1 top row compares the estimation of KL divergence with regular neural net and RKHS discriminator with complexity control based on eq. (13). With our proposed RKHS discriminator, the KL estimates are significantly more reliable and accurate: error reduced from 0.5 to 0.04, 5.8 to 1.07 and 60.6 to 9.7 and variance reduced from 0.2 to 0.002, 223 to 4.4 and 3521 to 33 for true KL 1.3, 13.8 and 38.29 respectively. In Fig. 1 bottom row, we investigate our complexity control method on the effect of varying the regularization parameter  $\lambda = \lambda_0 / m$ . As expected, increasing regularization parameter penalizes more on the RKHS norm and therefore reduces variance. This is consistent with our theory. Regarding bias, however, as we increase the  $\lambda$ , the bias decreases and then starts to increase. Hence, one needs to strike a balance between bias and variance while choosing  $\lambda$ .

Mutual Information Estimation Computation of mutual information is a direct use case of KL divergence computation. We replicate the experimental setup of [19, 9] to estimate mutual information between  $(x,y)$  drawn from 20-d Gaussian distributions, where the mutual information is increased by step size of 2 from 2 to 10. We compare the performance of our method with traditional KL divergence computation methods like contrastive predictive coding (CPC) [18], convex risk minimization (NWJ) [1] and SMILE [9]. In Fig.2, our method with RKHS discriminator (with  $\lambda = 1e^{-5}$ ) performs better than CPC [18] and NWJ [1], and is competitive with the state-of-the-art, SMILE [9]. In the bottom row, we also show the effect of regularization parameter  $\lambda$  in our method. Similar to the previous experiment, increasing the regularization parameter decreases the variance and increases the bias. It is consistent with our theoretical insights about the effect of reducing RKHS norm on variance.

Adversarial Variational Bayes Variational Bayes requires KL divergence estimation. When we do not have access to analytical form of the posterior/prior distributions, but only have access to the samples, we need to estimate KL divergence from samples. Adversarial Variational Bayes (AVB) [8] presents a way to achieve this using a discriminator network. We adopt this setup and demonstrate that the training becomes unstable if we do not constrain the complexity of the discriminator. First, we train AVB on MNIST dataset with a simple neural network discriminator architecture. As the training progresses, the KL divergence blows up after about 500 epochs (Fig. 3(b)) and the reconstruction starts to get worse (Fig. 3(a)). We modify the same architecture according to our construction such that it lies in RKHS and then penalize the RKHS norm as in eq. (13). It stabilizes the training for a large number of epochs and the reconstruction does not deteriorate as the training progresses, resulting into sharp reconstruction (Fig. 3(a)). We want to clarify that this instability in training

![](images/07ede3d17983d7ca6732fe60c04b30f5f8a778d25c4f93b5ca9eb98a8708ecfe.jpg)  
Neural Net discriminator

![](images/f10a2f88e8b109f90def96dccc5a1e3307051498471e8c4c787287ab2c9eed80.jpg)

![](images/e6dd230eead510ee85dab7889eca9bca5857077ff5e06d61e6d04d7296f88cae.jpg)

![](images/fde1b20acd73fc99877101e338e1215860d42153fe11fea2b1e42b557f1d2b58.jpg)

![](images/4fa09a9d5420d3dfc0461299381a2a5ae6565e86c338561b417a84f4caee28bf.jpg)

![](images/db18e95578608215b77153626f431b6fd9af9c189136b78280c9b50198d359f9.jpg)  
RKHS discriminator  
Epochs  
100

![](images/9d4e4e627b09a1b5dab98dcb4b11f7358d6e2c0a504a7337efd6fd3325f7ed04.jpg)  
300

![](images/39b9a6e4259d589d50b34831701c14aa5d0c99f4b2ac91482b17b600de3de7eb.jpg)  
(a)  
500

![](images/452c05ee9e003c55c8ff6d5e4700585b0a6961298454c4c6acc1c6a13dea5ecf.jpg)  
700

![](images/2820bf3712ceefb7d45b6427e2ed23fc1c25d804ad50f156cdc7b32d0a36b58c.jpg)  
900

![](images/ae006ab49bd87f1faec4caf96e3e4e2afdc90a69e8481526de8d4cc3b2c37f70.jpg)  
Neural Net discriminator  
(b)

![](images/9613f5c97a9db19a914120a5447c53ee05e211035fe4f74ff4ea9ad957c5026d.jpg)  
Figure 3: (a) Comparison of MNIST digit reconstruction using AVB autoencoder model [8]. Trace of KL divergence and reconstruction loss in AVB model with Neural network discriminator (b) and RKHS discriminator in (c).  
RKHS discriminator  
(c)

neural net discriminator is present if we use a basic discriminator architecture. It does not mean that there exists no other method to design a stable neural net discriminator. In fact, AVB [8] presents a discriminator that adds additional inner product structure to stabilize the discriminator training. Our point here is that we can stabilize the training by ensuring that the discriminator lies in a well behaved function space (the RKHS) and controlling its complexity, consistent with our theory.

# 9 Limitations, Discussion and Conclusion

Limitations: The proposed construction of neural function in RKHS exhibits good properties of both the deep learning and kernel methods. However, it requires constructing two separate deep networks,  $\psi$  and  $g$ . It makes our model a bit bulky and also requires more parameter due to additional  $g$ . Moreover, currently our RKHS discriminator's output is scalar; generalizing this function to a multivariable output could make our model bulkier and increase parameters even more. Second limitation is the requirement of higher order derivative of kernel  $K$  in assumption A3. While this requirement is satisfied if smooth activation function is used in  $\phi_{\theta}$ , for activations like ReLU or LeakyReLU, the derivatives exist everywhere except at the origin. In these cases, we need to carefully investigate if we can use subgradients to define operator norm  $||\mathcal{L}_p||$ .

Discussion and Conclusion: We have shown that using a regular neural network as a discriminator in estimating KL divergence results in unreliable estimation if the complexity of the function space is not controlled. We then showed a solution by constructing a discriminator function in RKHS space using neural networks and penalizing its complexity in a scalable way. Although the idea to use RKHS norm to penalize complexity is not new (see for example [1]), it is not clear how to use this idea directly on the function  $f$ . In traditional kernel methods, algorithms often do not work with RKHS function  $f$  directly, but rather work with kernel matrix,  $K$  by using, for example, the Representer Theorem [20]. In the case of big data, working with the big kernel matrix is computationally expensive although some methods have been proposed to speed up the computation, like Random Fourier Feature [21]. We propose a different view by directly constructing a function in RKHS space, which led us to scalable algorithm while incorporating the advantages of neural networks. Moreover, our representation could also be seen as an improvement over RFF by using neural basis,  $\psi$ , instead of Fourier basis. The idea of constructing a neural-net function in RKHS and complexity control could also be useful in stabilizing GANs in general. Currently, the most successful way to stabilize GANs is to enforce smoothness by gradient penalization [22, 23, 24]. On the light of the present analysis, gradient penalty could also be thought as a way to control the complexity of the discriminator.

# References

[1] X. Nguyen, M. J. Wainwright, and M. I. Jordan, "Estimating divergence functionals and the likelihood ratio by convex risk minimization," IEEE Transactions on Information Theory, vol. 56, no. 11, pp. 5847-5861, 2010.  
[2] T. Q. Chen, X. Li, R. B. Grosse, and D. K. Duvenaud, "Isolating sources of disentanglement in variational autoencoders," in Advances in Neural Information Processing Systems, pp. 2610-2620, 2018.  
[3] L. Mescheder, A. Geiger, and S. Nowozin, "Which training methods for gans do actually converge?", in International Conference on Machine Learning, pp. 3481-3490, 2018.  
[4] X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel, “Infogan: Interpretable representation learning by information maximizing generative adversarial nets,” in Advances in neural information processing systems, pp. 2172–2180, 2016.  
[5] C. K. Sønderby, J. Caballero, L. Theis, W. Shi, and F. Huszár, "Amortised map inference for image super-resolution," ICLR, 2017.  
[6] S. Nowozin, B. Cseke, and R. Tomioka, "f-gan: Training generative neural samplers using variational divergence minimization," in Advances in neural information processing systems, pp. 271-279, 2016.  
[7] M. I. Belghazi, A. Baratin, S. Rajeshwar, S. Ozair, Y. Bengio, A. Courville, and D. Hjelm, "Mutual information neural estimation," in International Conference on Machine Learning, pp. 531-540, 2018.  
[8] L. Mescheder, S. Nowozin, and A. Geiger, "Adversarial variational bayes: Unifying variational autoencoders and generative adversarial networks," in International Conference on Machine Learning (ICML), 2017.  
[9] J. Song and S. Ermon, "Understanding the limitations of variational mutual information estimators," in International Conference on Learning Representations, 2020.  
[10] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding deep learning requires rethinking generalization,” arXiv preprint arXiv:1611.03530, 2016.  
[11] K. Ahuja, "Estimating kullback-leibler divergence using kernel machines," in 2019 53rd Asilomar Conference on Signals, Systems, and Computers, pp. 690-696, IEEE, 2019.  
[12] A. Berlinet and C. Thomas-Agnan, Reproducing Kernel Hilbert Spaces in Probability and Statistics. Springer US, 2011.  
[13] F. Bach, “On the equivalence between kernel quadrature rules and random feature expansions,” The Journal of Machine Learning Research, vol. 18, no. 1, pp. 714–751, 2017.  
[14] F. Cucker and S. Smale, “On the mathematical foundations of learning,” Bulletin of the American mathematical society, vol. 39, no. 1, pp. 1–49, 2002.  
[15] F. Bach, “Breaking the curse of dimensionality with convex neural networks,” The Journal of Machine Learning Research, vol. 18, no. 1, pp. 629–681, 2017.  
[16] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in International Conference on Learning Representations, 2018.  
[17] A. W. Van Der Vaart and J. A. Wellner, “Weak convergence,” in Weak convergence and empirical processes, Springer, 1996.  
[18] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.  
[19] B. Poole, S. Ozair, A. Van Den Oord, A. Alemi, and G. Tucker, "On variational bounds of mutual information," in International Conference on Machine Learning, pp. 5171-5180, PMLR, 2019.  
[20] B. Schölkopf, A. J. Smola, F. Bach, et al., Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press, 2002.  
[21] A. Rahimi, B. Recht, et al., “Random features for large-scale kernel machines,” in Neural Information Processing Systems, vol. 3, p. 5, Citeseer, 2007.

[22] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in International Conference on Machine Learning, pp. 214-223, 2017.  
[23] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, "Improved training of Wasserstein gans," in Advances in Neural Information Processing Systems 30 (I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, eds.), pp. 5767-5777, Curran Associates, Inc., 2017.  
[24] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD GANs,” in International Conference on Learning Representations, 2018.
