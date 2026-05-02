# LANGEVIN AUTOENCODERS FOR LEARNING DEEP LATENT VARIABLE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Markov chain Monte Carlo (MCMC), such as Langevin dynamics, is valid for approximating intractable distributions. However, its usage is limited in the context of deep latent variable models since it is not scalable to data size owing to its datapoint-wise iterations and slow convergence. This paper proposes the amortized Langevin dynamics (ALD), wherein datapoint-wise MCMC iterations are entirely replaced with updates of an inference model that maps observations into latent variables. Since it no longer depends on datapoint-wise iterations, ALD enables scalable inference from large-scale datasets. Despite its efficiency, it retains the excellent property of MCMC; we prove that ALD has the target posterior as a stationary distribution with a mild assumption. Furthermore, ALD can be extended to sampling from an unconditional distribution such as an energy-based model, enabling more flexible generative modeling by applying it to the prior distribution of the latent variable. Based on ALD, we construct a new deep latent variable model named the Langevin autoencoder (LAE). LAE uses ALD for autoencoder-like posterior inference and sampling from the latent space EBM. Using toy datasets, we empirically validate that ALD can properly obtain samples from target distributions in both conditional and unconditional cases, and ALD converges significantly faster than traditional LD. We also evaluate LAE on the image generation task using three datasets (SVHN, CIFAR-10, and CelebA-HQ). Not only can LAE be trained faster than non-amortized MCMC methods, but LAE can also generate better samples in terms of the Fréchet Inception Distance (FID) compared to AVI-based methods, such as the variational autoencoder<sup>1</sup>.

# 1 INTRODUCTION

Variational inference (VI) and Markov chain Monte Carlo (MCMC) are two practical tools to approximate intractable distributions. Recently, VI has been dominantly used in deep latent variable models (DLVMs) to approximate the posterior distribution over the latent variable  $\mathbf{z}$  given the observation  $\mathbf{x}$ , i.e.,  $p(\mathbf{z} \mid \mathbf{x})$ . At the core of the success of VI is the invention of amortized variational inference (AVI) (Kingma et al., 2014; An & Cho, 2015; Su et al., 2018; Eslami et al., 2018; Kumar et al., 2018), which replaces optimization of datapoint-wise variational parameters with an inference model that predicts latent variables from observations. This amortization enables posterior inference to be performed efficiently on large-scale datasets, even for new observations. However, the approximation power of AVI (or VI itself) is limited because it relies on distributions with tractable densities for approximations. Although there have been attempts to improve their flexibility (e.g., normalizing flows (Rezende & Mohamed, 2015; Kingma et al., 2016; Van Den Berg et al., 2018; Huang et al., 2018)), such methods typically have architectural constraints (e.g., invertibility in normalizing flows).

Compared to VI, MCMC can approximate complex distributions because it does not rely on any tractable distributions. Instead, MCMC repeats sampling from the target distribution and uses obtained samples to approximate the posterior distribution. Langevin dynamics is a typical example of MCMC for sampling from a continuous distribution. However, despite its high approximation ability, MCMC has received relatively little attention in learning DLVMs. It is because MCMC methods take a long time to converge, making it difficult to be used in the training of DLVMs. When learning

![](images/5f848d16fd0e2bae0aa13b66b908fffaa959af83d42d5d301aab862171543a05.jpg)  
(A)

![](images/32edb665eb6aba224781ac887d36c13f4b716a48f3cbb4c03c3163ce3de92d8a.jpg)  
(B-1) Traditional LD

![](images/6f97ab2b6588e63df3b57845c8457ca4953700a13084124b87027b15bfd87573.jpg)  
Figure 1: (A) Directed graphical model under consideration. (B-1) In traditional Langevin dynamics, the samples are directly updated in the latent space. (B-2) Our amortized Langevin dynamics replace the update of latent samples with an inference model  $f_{\mathbf{z}|\mathbf{x}}$  that maps the observation  $\mathbf{x}$  into the latent variable  $\mathbf{z}$ .  
(B-2) Proposed Method (ALD)

DLVMs with MCMC, we need to run MCMC iterations for sampling from each posterior per data point, i.e.,  $p(\mathbf{z} \mid \mathbf{x}^{(i)})$  ( $i = 1, \dots, n$ ), where  $n$  is the number of training data, as shown in Figure 1 (B-1). It is problematic, mainly when training with a large-scale ( $n > 10\mathrm{K}$ ) dataset, because it is time-consuming to run massive MCMC iterations for all data points. Furthermore, we need to re-run the sampling procedure when we obtain new observations in test time.

As in VI, there have been some attempts to introduce the concept of amortized inference to MCMC. For example, Hoffman (2017) initializes MCMC sampling using an inference model that predicts latent variables from observations. However, as they use inference models only for the initialization of MCMC, these methods still rely on datapoint-wise sampling iterations. Not only is it time-consuming, but implementations of such partially amortized methods also tend to be complicated compared to the simplicity of AVI. To make MCMC more suitable for the inference of DLVMs, a more straightforward and sophisticated framework of amortization is needed.

This paper proposes the amortized Langevin dynamics (ALD), which replace datapoint-wise MCMC iterations with updates of an inference model that maps observations into latent variables (Figure 1 (B-2)). Since latent variables depend on the inference model, the updates of the inference model can be regarded as implicit updates of latent variables, which enables us to perform posterior inference without datapoint-wise MCMC iterations. Notably, our ALD treats outputs of the inference model themselves as samples from the target distribution, whereas existing amortization methods use the outputs only as initialization of MCMC. Therefore, ALD can be implemented straightforwardly like AVI. Moreover, despite the simplicity, we can theoretically guarantee that ALD has the true posterior as a stationary distribution under a mild assumption, which is a critical requirement for valid MCMC algorithms.

Although we have introduced ALD as a posterior sampling algorithm, the application of ALD is not limited to sampling from posterior distributions. Recent studies have demonstrated that applying an energy-based model (EBM) into the prior distribution over the latent variable enables more flexible generative modeling for DLVMs. When we train an EBM, we have to obtain samples from the EBM by running costly MCMC iterations. By extending our ALD to sampling from unconditional distributions, we can apply ALD into sampling from such EBMs. When sampling from an EBM with ALD, we prepare a function that maps fixed inputs into latent variables, and updates of EBM samples are replaced with updates of the function's parameters. In the same way with the posterior case, the updates of the sampler function can be regarded as implicit updates of samples from EBMs.

Using our ALD for sampling from both the posterior and the EBM over the latent variable, we derive a novel framework of learning DLVMs, which we refer to as the Langevin autoencoder (LAE). Interestingly, the learning algorithm of LAE naturally takes the combined form of an autoencoder-like architecture and adversarial training. Our experiments show that ALD can properly obtain samples from target distributions using toy datasets. Subsequently, we perform numerical experiments of the image generation task using the SVHN, CIFAR-10, and CelebA-HQ datasets. Not only can LAE

be trained faster than non-amortized MCMC methods, but LAE can also generate better samples in terms of the Fréchet Inception Distance compared to AVI-based methods, such as VAE.

# 2 PRELIMINARIES

# 2.1 PROBLEM DEFINITION

Consider a probabilistic model with the  $d_{\mathbf{x}}$ -dimensional observation  $\mathbf{x}$ , the  $d_{\mathbf{z}}$ -dimensional continuous latent variable  $\mathbf{z}$ , and the model parameter  $\Theta$ , as described by the probabilistic graphical model shown in Figure 1 (A). Although the posterior distribution over the latent variable is proportional to the prior and the likelihood:  $p(\mathbf{z} \mid \mathbf{x}) = p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z}) / p(\mathbf{x})$ , this is intractable due to the normalizing constant  $p(\mathbf{x}) = \int p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z}) d\mathbf{z}$ . This study aims to approximate the posterior  $p(\mathbf{z} \mid \mathbf{x})$  for all  $n$  observations  $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$  efficiently by obtaining samples from it.

# 2.2 LANGEVIN DYNAMICS

Langevin dynamics (LD) (Neal, 2011) is a sampling algorithm based on the following Langevin equation:

$$
d \boldsymbol {z} = - \nabla_ {\boldsymbol {z}} U (\boldsymbol {x}, \boldsymbol {z}) d t + \sqrt {2 \beta^ {- 1}} d B, \tag {1}
$$

where  $U$  is a Lipschitz continuous potential function that satisfies an appropriate growth condition,  $\beta$  is an inverse temperature parameter, and  $B$  is a Brownian motion. This stochastic differential equation has  $p^{\beta}(\mathbf{z}\mid \mathbf{x})\propto \exp (-\beta U(\mathbf{x},\mathbf{z}))$  as its equilibrium distribution. We set  $\beta = 1$  and define the potential as follows to obtain the target posterior  $p(\mathbf{z}\mid \mathbf{x})$  as its equilibrium:

$$
U (\boldsymbol {x}, \boldsymbol {z}) = - \log p (\boldsymbol {z}) - \log p (\boldsymbol {x} \mid \boldsymbol {z}). \tag {2}
$$

We can obtain samples from the posterior by simulating Eq. (1) using the Euler-Maruyama method (Kloeden & Platen, 2013) as follows:

$$
\boldsymbol {z} ^ {\prime} \sim \mathcal {N} \left(\boldsymbol {z} ^ {\prime}; \boldsymbol {z} - \eta \nabla_ {z} U (\boldsymbol {x}, \boldsymbol {z}), 2 \eta \boldsymbol {I}\right), \tag {3}
$$

where  $\eta$  is a step size for discretization. When the step size is sufficiently small, the samples asymptotically move to the target posterior by repeating this sampling iteration. LD can be applied to any posterior inference problems for continuous latent variables, provided the potential energy is differentiable on the latent space. However, to obtain the posterior samples for all observations  $\pmb{x}^{(1)},\dots \pmb{x}^{(n)}$ , we should perform iterations of Eq. (3) per data point, as shown in Figure 1 (B-1). It is inefficient, mainly if the dataset is large. In addition, we need to re-run the time-consuming iterations for new observations in test time. In the next section, we demonstrate a method that addresses the inefficiency by amortization.

# 3 AMORTIZED LANGEVIN DYNAMICS

# 3.1 GENERAL IDEA

As an alternative to the direct simulation of latent dynamics, we define an inference model  $f_{\mathbf{z}|\mathbf{x}}$  which maps the observation into the latent variable. Formally, the dynamics of its parameter  $\Phi$  is

$$
d \boldsymbol {\Phi} = - \sum_ {i = 1} ^ {n} \nabla_ {\boldsymbol {\Phi}} U \left(\boldsymbol {x} ^ {(i)}, f _ {\mathbf {z} | \mathbf {x}} \left(\boldsymbol {x} ^ {(i)}; \boldsymbol {\Phi}\right)\right) d t + \sqrt {2} d B. \tag {4}
$$

Because the function  $f_{\mathbf{z}|\mathbf{x}}$  outputs latent variables, the stochastic dynamics on the parameter space induce dynamics on the latent space. The main idea of our amortized Langevin dynamics (ALD) is to regard the transition on this induced dynamics as a sampling procedure for the posterior distributions, as shown in Figure 1 (B-2).

We can use the Euler-Maruyama method to simulate Eq. (4) like traditional LD:

$$
\boldsymbol {\Phi} ^ {\prime} \sim \mathcal {N} \left(\boldsymbol {\Phi} ^ {\prime}; \boldsymbol {\Phi} - \eta \sum_ {i = 1} ^ {n} \nabla_ {\boldsymbol {\Phi}} U \left(\boldsymbol {x} ^ {(i)}, f _ {\mathbf {z} | \mathbf {x}} \left(\boldsymbol {x} ^ {(i)}; \boldsymbol {\Phi}\right)\right), 2 \eta \boldsymbol {I}\right). \tag {5}
$$

Algorithm 1 Amortized Langevin dynamics  
$\Phi \leftarrow$  Initialize parameters   
 $\mathbb{Z}^{(1)},\ldots ,\mathbb{Z}^{(n)}\gets \emptyset$  ▷ Initialize sample sets for all  $n$  data points   
repeat  $\Phi \leftarrow \Phi^{\prime}\sim \mathcal{N}\left(\Phi^{\prime};\Phi -\eta \sum_{i = 1}^{n}\nabla_{\Phi}U\left(\boldsymbol{x}^{(i)},\boldsymbol{z}^{(i)} = f_{\mathbf{z}|\mathbf{x}}\left(\boldsymbol{x}^{(i)};\Phi\right)\right),2\eta_{\phi}\boldsymbol {I}\right)$ $\mathbb{Z}^{(1)},\ldots ,\mathbb{Z}^{(n)}\gets \mathbb{Z}^{(1)}\cup \left\{f_{\mathbf{z}|\mathbf{x}}\left(\boldsymbol{x}^{(1)};\Phi\right)\right\} ,\ldots ,\mathbb{Z}^{(N)}\cup \left\{f_{\mathbf{z}|\mathbf{x}}\left(\boldsymbol{x}^{(n)};\Phi\right)\right\}$  ▷ Add samples   
until convergence of parameters   
return  $\mathbb{Z}^{(1)},\ldots ,\mathbb{Z}^{(n)}$

Through the iterations, the posterior sampling is implicitly performed by collecting outputs of the inference model for each data point as described in Algorithm 1. Note that  $\mathbb{Z}^{(i)}$  denotes a set of samples of the posterior for the  $i$ -th data (i.e.,  $p(\mathbf{z} \mid \mathbf{x}^{(i)})$ ) obtained using ALD. When we perform inference for new test data, the trained inference model can be used to initialize an MCMC method (e.g., traditional LD) because it is expected that the trained inference model can map data into the high-density area of the posteriors.

By this amortization, we replace the direct update of latent variables  $(z^{(1)},\dots,z^{(n)})$  with the update of the global parameter  $\Phi$ . A significant advantage of amortization is that the cost of MCMC can be reduced by using minibatch training. For minibatch training, we substitute the minibatch statistics of  $m$  data points for the derivative for all  $n$  data.

$$
\sum_ {i = 1} ^ {n} \nabla_ {\Phi} U \left(\boldsymbol {x} ^ {(i)}, f _ {\mathbf {z} | \mathbf {x}} \left(\boldsymbol {x} ^ {(i)}; \Phi\right)\right) \approx \frac {n}{m} \sum_ {i = 1} ^ {m} \nabla_ {\phi} U \left(\boldsymbol {x} ^ {(i)}, f _ {\mathbf {z} | \mathbf {x}} \left(\boldsymbol {x} ^ {(i)}\right)\right).
$$

We refer to the minibatch version of ALD as stochastic gradient amortized Langevin dynamics (SGALD). SGALD enables us to sample from posteriors of a massive dataset efficiently. Moreover, in the context of stochastic gradient LD (SGLD), it is known that adaptive preconditioning effectively improves convergence compared to the naive SGLD (Li et al., 2016)<sup>2</sup>. This preconditioning technique is also applicable to our SGALD, and we employ it throughout our experiments.

# 3.2 THEORETICAL ANALYSIS

To justify our ALD as a posterior sampling algorithm, we provide a theoretical analysis of the stationary distribution of our ALD algorithm. Here,  $\mathbf{X}$  and  $\mathbf{Z}$  denote matrices with  $\mathbf{x}^{(i)}$  and  $\mathbf{z}^{(i)}$  in rows  $\mathbf{X}_{i,:}$  and  $\mathbf{Z}_{i,:}$ , respectively. Our main result is as follows:

Theorem 1. Let  $q(\mathbf{Z} \mid \mathbf{X})$  be a stationary distribution of the latent variables induced by Eq. (4). When the mapping  $f_{\mathbf{z}|\mathbf{x}}$  meets the following conditions,  $q(\mathbf{Z} \mid \mathbf{X})$  satisfies  $q(\mathbf{Z} \mid \mathbf{X}) \propto \exp(-U(\mathbf{X},\mathbf{Z})) \coloneqq \exp\left(-\sum_{i=1}^{n} U(\mathbf{x}^{(i)},\mathbf{z}^{(i)})\right)$ .

1. The mapping has the form of  $f_{\mathbf{z}|\mathbf{x}}(\mathbf{x}; \Phi) = \Phi g(\mathbf{x})$ , where  $\Phi$  is a  $d_{\mathbf{z}} \times d$  matrix,  $g$  is a mapping from  $\mathbb{R}^{d_{\mathbf{x}}}$  to  $\mathbb{R}^d$ , and  $d$  is the dimensionality of  $g(\mathbf{x})$ .  
2. The rank of  $G$  is  $n$ , where  $G$  is a matrix with  $g(\boldsymbol{x}^{(i)})$  in row  $G_{i,..}$ .

See Appendix A for the proof. Theorem 1 suggests that samples obtained by ALD asymptotically converge to the true posterior when we construct the inference model  $f_{\mathbf{z}|\mathbf{x}}$  with an appropriate form. Practically, we can implement such a function using a neural network whose parameters are fixed except for the last linear layer. In this implementation, the last linear layer takes a role of the parameter  $\Phi$ , and the preceding feature extractor takes a role of the function  $g$ . In addition, the dimensionality of the last linear layer should be sufficiently large to meet the second condition. Therefore, the second condition derives a trade-off between approximation quality and computational costs. It is worth noting that a similar trade-off is also known in the context of AVI. In AVI, the approximation

quality is influenced by the capacity of the inference model, and the gap between the optimal variational distribution and the amortized distribution is often denoted as the amortization gap (Cremer et al., 2018). In experiments, we confirm that preserving the condition does not become a significant computational overhead in practice. In addition, we should note that decaying the step size  $\eta$  is needed to ensure convergence when we perform the simulation with discretization as described by Welling & Teh (2011).

# 3.3 EXTENSION TO UNCONDITIONAL CASES

Currently, we have introduced ALD as a sampling algorithm for conditional posterior distributions. We can also apply ALD to sampling from unconditional unnormalized distributions, namely energy-based models (EBMs). Consider an unconditional distribution over a random variable  $\mathbf{z}$  defined by an energy function  $f_{\mathbf{z}}$  as follows:

$$
p (\boldsymbol {z}) \propto \exp (- f _ {\mathbf {z}} (\boldsymbol {z})) ， \tag {6}
$$

where  $f_{\mathbf{z}}$  maps the variable  $z$  into a scalar value. To obtain samples from this EBM using ALD, we prepare a sampler function  $f_{\mathbf{z}|\mathbf{u}}(\mathbf{u};\boldsymbol{\Psi})$  that maps its input  $\mathbf{u}$  into the variable  $z$ . Here, the input vector  $\mathbf{u}$  is fixed, whereas observations are used in the posterior case. To run multiple MCMC chains in parallel, we prepare  $k$  fixed inputs  $\mathbf{u}^{(1)},\ldots ,\mathbf{u}^{(k)}$  and update the parameter of the sampler function as follows:

$$
\Psi^ {\prime} \sim \mathcal {N} \left(\Psi - \eta \sum_ {i = 1} ^ {k} \nabla_ {\Psi} f _ {\mathbf {z}} \left(f _ {\mathbf {z} | \mathbf {u}} \left(\boldsymbol {u} ^ {(i)}; \Psi\right)\right), 2 \eta I\right). \tag {7}
$$

Typically, the fixed input vectors  $\mathbf{u}^{(1)},\ldots ,\mathbf{u}^{(k)}$  are chosen from a standard Gaussian distribution. As in the posterior case, we can guarantee the stationary distribution matches the EBM by choosing an appropriate form for the function  $f_{\mathbf{z}|\mathbf{u}}$ , and such a function can be implemented using a neural network whose parameters are fixed except for the last linear layer. For minibatch training, we can substitute the gradient for all  $k$  chains with the stochastic gradient of  $m$  minibatch chains:

$$
\sum_ {i = 1} ^ {k} \nabla_ {\Psi} f _ {\mathbf {z}} \left(f _ {\mathbf {z} | \mathbf {u}} \left(\boldsymbol {u} ^ {(i)}; \boldsymbol {\Psi}\right)\right) \approx \frac {k}{m} \sum_ {i = 1} ^ {m} \nabla_ {\Psi} f _ {\mathbf {z}} \left(f _ {\mathbf {z} | \mathbf {u}} \left(\boldsymbol {u} ^ {(i)}; \boldsymbol {\Psi}\right)\right). \tag {8}
$$

The advantage of using amortization in the unconditional case is that we can run massive chains parallel using minibatch training.

# 4 LANGEVIN AUTOENCODERS

Using ALD for sampling from both the posterior and the energy-based prior, we derive a novel framework for learning DLVMs. We here consider a latent variable model defined as follows:

$$
p \left(\boldsymbol {z} \mid \boldsymbol {\Theta}\right) \propto \exp \left(- f _ {\mathbf {z}} (\boldsymbol {z}; \boldsymbol {\Theta})\right), p \left(\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {\Theta}\right) = \mathcal {N} \left(\boldsymbol {x}; f _ {\mathbf {x} | \mathbf {z}} (\boldsymbol {z}; \boldsymbol {\Theta}), \operatorname {d i a g} (\boldsymbol {\sigma})\right), \tag {9}
$$

where  $\sigma$  is a variance parameter of a Gaussian distribution<sup>3</sup>. To learn the latent variable model, we need to estimate both the model parameter  $\Theta$  and the latent variable  $\mathbf{z}$ . In Bayesian learning, the estimation is represented as the joint posterior distribution  $p\left(\Theta, \mathbf{z}^{(1)}, \ldots, \mathbf{z}^{(n)} \mid \mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right)$ . To obtain samples from the posterior, we can combine traditional LD and ALD as follows:

$$
\boldsymbol {\Theta} ^ {\prime} \sim \mathcal {N} \left(\boldsymbol {\Theta} ^ {\prime}; \boldsymbol {\Theta} - \eta \nabla_ {\boldsymbol {\Theta}} U \left(\boldsymbol {X}, f _ {\mathbf {z} | \mathbf {x}} (\boldsymbol {X}; \boldsymbol {\Phi}), \boldsymbol {\Theta}\right), 2 \eta \boldsymbol {I}\right), \tag {10}
$$

$$
\boldsymbol {\Phi} ^ {\prime} \sim \mathcal {N} \left(\boldsymbol {\Phi} ^ {\prime}; \boldsymbol {\Phi} - \eta \nabla_ {\boldsymbol {\Phi}} U \left(\boldsymbol {X}, f _ {\mathbf {z} | \mathbf {x}} (\boldsymbol {X}; \boldsymbol {\Phi}), \boldsymbol {\Theta}\right), 2 \eta \boldsymbol {I}\right), \tag {11}
$$

$$
U \left(\boldsymbol {X}, \boldsymbol {Z}, \boldsymbol {\Theta}\right) := - \log p (\boldsymbol {\Theta}) - \sum_ {i = 1} ^ {n} \log p \left(\boldsymbol {z} ^ {(i)} \mid \boldsymbol {\Theta}\right) + \log p \left(\boldsymbol {x} ^ {(i)} \mid \boldsymbol {z} ^ {(i)}, \boldsymbol {\Theta}\right), \tag {12}
$$

where  $f_{\mathbf{z}|\mathbf{x}}(\mathbf{X};\Phi)$  is a matrix with  $f_{\mathbf{z}|\mathbf{x}}(\pmb{x}^{(i)};\Phi)$  in its  $i$ -th row. If we omit the Gaussian noise injection in Eq. (10), it corresponds to gradient ascent for maximum a posteriori (MAP) estimation

# Algorithm 2 Langevin Autoencoders

$\Theta ,\Phi ,\Psi \leftarrow$  Initialize parameters

repeat

$$
\boldsymbol {\Theta} \leftarrow \boldsymbol {\Theta} ^ {\prime} \sim \mathcal {N} \left(\boldsymbol {\Theta} ^ {\prime}; \boldsymbol {\Theta} - \eta \nabla_ {\boldsymbol {\Theta}} \mathcal {L} (\boldsymbol {\Theta}, \boldsymbol {\Phi}, \boldsymbol {\Psi}), 2 \eta \boldsymbol {I}\right)
$$

$$
\boldsymbol {\Phi} \leftarrow \boldsymbol {\Phi} ^ {\prime} \sim \mathcal {N} \left(\boldsymbol {\Phi} ^ {\prime}; \boldsymbol {\Phi} - \eta \nabla_ {\boldsymbol {\Phi}} \mathcal {L} (\boldsymbol {\Theta}, \boldsymbol {\Phi}, \boldsymbol {\Psi}), 2 \eta \boldsymbol {I}\right)
$$

$$
\Psi \leftarrow \Psi^ {\prime} \sim \mathcal {N} \left(\Psi^ {\prime}; \Psi + \eta \nabla_ {\Psi} \mathcal {L} (\Theta , \Phi , \Psi), 2 \eta I\right)
$$

until convergence of parameters

return  $\Theta ,\Phi ,\Psi$

$\triangleright$  Update the generative model

$\triangleright$  Update the inference model

> Update the sampler model

of  $\Theta$ ; if we additionally use a flat prior for  $p(\Theta)$ , it yields the maximum likelihood estimation (MLE). In this study, we assume a flat prior for  $p(\Theta)$  and omit the notation for simplicity.

In Eq. (10), we cannot calculate the derivative of the potential function  $\nabla_{\Theta}U$  in a closed-form because the latent prior  $p(z|\Theta)$  is defined using an unnormalized energy function. However, we can obtain the unbiased estimator of the derivative by obtaining samples from the prior as follows:

$$
\begin{array}{l} \nabla_ {\Theta} U (\boldsymbol {X}, \boldsymbol {Z}, \Theta) \\ \approx \sum_ {i = 1} ^ {n} \nabla_ {\Theta} f _ {\mathbf {z}} \left(\boldsymbol {z} ^ {(i)}; \boldsymbol {\Theta}\right) - \nabla_ {\Theta} \log p \left(\boldsymbol {x} ^ {(i)} \mid \boldsymbol {z} ^ {(i)}, \boldsymbol {\Theta}\right) - \frac {n}{k} \sum_ {j = 1} ^ {k} \nabla_ {\Theta} f _ {\mathbf {z}} \left(\tilde {\boldsymbol {z}} ^ {(j)}; \boldsymbol {\Theta}\right), \tag {13} \\ \end{array}
$$

where  $\tilde{z}^{(1)},\ldots ,\tilde{z}^{(n)}$  are sampled from the latent prior  $p(\mathbf{z}\mid \Theta)$  (see Appendix B for the derivation). To get samples from the latent prior, we can also use ALD as described in Section 3.3. Here, we set the number of chains equal to the number of data points for simplicity, i.e.,  $k = n$ .

In summary, the encoder  $f_{\mathbf{z}|\mathbf{x}}$ , the decoder  $f_{\mathbf{x}|\mathbf{z}}$ , and the latent energy function  $f_{\mathbf{z}}$  are trained by minimizing the following loss function  $\mathcal{L}$ , whereas the latent sampler  $f_{\mathbf{z}|\mathbf{u}}$  is trained by maximizing it, while stochastic noise of the Brownian motion is injected in their update in order to avoid shrinking to MAP estimates:

$$
\begin{array}{l} \mathcal {L} (\Theta , \Phi , \Psi) \tag {14} \\ = \sum_ {i = 1} ^ {n} f _ {\mathbf {z}} \left(f _ {\mathbf {z} | \mathbf {x}} \left(\boldsymbol {x} ^ {(i)}; \boldsymbol {\Phi}\right); \boldsymbol {\Theta}\right) - f _ {\mathbf {z}} \left(f _ {\mathbf {z} | \mathbf {u}} \left(\boldsymbol {u} ^ {(i)}; \boldsymbol {\Psi}\right); \boldsymbol {\Theta}\right) - \log p \left(\boldsymbol {x} ^ {(i)} \mid f _ {\mathbf {z} | \mathbf {x}} \left(\boldsymbol {x} ^ {(i)}; \boldsymbol {\Phi}\right), \boldsymbol {\Theta}\right). \\ \end{array}
$$

We refer to this framework of learning DLVMs as the Langevin autoencoder (LAE). We summarize the algorithm of the LAE in Algorithm 2. LAE is closely related to the traditional autoencoder (AE) and other deep generative models, such as the variational autoencoder (VAE) and the generative adversarial network (GAN). We discuss the relationship in the next section in detail.

# 5 RELATED WORKS

In this section, we put our work within the bigger picture of deep generative models. Behind the recent success of deep generative models, there have been significant improvements in the neural network architecture and the learning algorithm. We here focus on the latter because our main focus in this paper is on the algorithmic aspect rather than specific neural network architectures.

Amortized inference is well-investigated in the context of variational inference. It is often referred to as amortized variational inference (AVI) (Rezende & Mohamed, 2015; Shu et al., 2018). The basic idea of AVI is to replace the optimization of the datapoint-wise variational parameters with the optimization of shared parameters across all data points by introducing an inference model that predicts latent variables from observations. The AVI is commonly used in generative models (Kingma & Welling, 2013), semi-supervised learning (Kingma et al., 2014), anomaly detection (An & Cho, 2015), machine translation (Su et al., 2018), and neural rendering (Eslami et al., 2018; Kumar et al., 2018). However, in the MCMC literature, there are few works on such amortization. Han et al. (2017) use traditional LD to obtain samples from posteriors to train deep latent variable models. Such Langevin-based algorithms for deep latent variable models are called alternating

back-propagation (ABP) and are applied in several fields (Xie et al., 2019; Zhang et al., 2020; Xing et al., 2018; Zhu et al., 2019). However, ABP requires datapoint-wise Langevin iterations, causing slow convergence. Moreover, when we perform inference for new data in test time, ABP requires MCMC iterations from randomly initialized samples again. Although Li et al. (2017) and Hoffman (2017) propose amortization methods for MCMC, they only amortize the initialization cost in MCMC by using an inference model. Therefore, they do not entirely remove datapoint-wise MCMC iterations.

Autoencoders (AEs) (Hinton & Salakhutdinov, 2006) can be seen as a particular case of LAEs, wherein the Gaussian noise injection to the update of the inference model (encoder) and the generative model (decoder) is omitted in Eqs. (10) and (11), and a flat prior is used for  $p(\mathbf{z} \mid \boldsymbol{\Theta})$ . When a different distribution is used as a latent prior, it is known as sparse autoencoders (SAEs) (Ng et al., 2011). In these cases, the dynamics in Eqs. (10) and (11) are dominated by gradient  $\nabla U$ ; hence, both the latent variables and the model parameter converge to MLE or MAP estimates (or other stationary points). Therefore, AEs (and SAEs) can be considered MLE (and MAP) algorithms for the parameter  $\boldsymbol{\Theta}$  and the latent variables  $\mathbf{Z}$ .

Variational Autoencoders (VAEs) are based on AVI, wherein an inference model (encoder) is defined as a variational distribution  $q(\mathbf{z} \mid \mathbf{x}; \boldsymbol{\Phi})$  using a neural network. Its parameter  $\boldsymbol{\Phi}$  is optimized by maximizing the evidence lower bound. Interestingly, there is a contrast between VAE and LAE when stochastic noise is used in posterior inference. In VAE, noise is used to sample from the stochastic inference model in calculating the potential  $U$ , i.e., in the forward calculation. However, in LAE, the inference model itself is deterministic, and stochastic noise is used for its parameter update along with the gradient calculation  $\nabla_{\phi} U$ , i.e., in the backward calculation. The advantage of LAE over VAE is that LAE can flexibly approximate complex posteriors by obtaining samples, whereas VAE's approximation ability is limited by choice of variational distribution  $q(\mathbf{z} \mid \mathbf{x}; \boldsymbol{\Phi})$  because it requires a tractable density function. Although there are several considerations in the improvement of the approximation flexibility, these methods typically have architectural constraints (e.g., invertibility and ease of Jacobian calculation in normalizing flows (Rezende & Mohamed, 2015; Kingma et al., 2016; Van Den Berg et al., 2018; Huang et al., 2018; Titsias & Ruiz, 2019)), or they incur more computational costs (e.g., MCMC sampling for the reverse conditional distribution in unbiased implicit variational inference (Titsias & Ruiz, 2019)).

Energy-based Models' training is challenging, and many researchers have been studying methodology for its stable and practical training. A significant challenge is that it requires MCMC sampling from EBMs, which is challenging to perform in high dimensional space. Our LAE avoids this difficulty by defining the energy function in latent space rather than data space. A similar approach is taken in several works (Pang et al., 2020a;b), but they use traditional LD to obtain latent samples without amortization.

Generative adversarial networks (GANs) are closely related to our LAE because both are trained using adversarial loss functions. For a detailed discussion, see Appendix C.

# 6 EXPERIMENT

In our experiment, we first test our ALD algorithm on toy examples to investigate its behavior, then we show the results of its application to the training of deep generative models.

# 6.1 TOY EXAMPLES

We perform numerical simulation using toy examples to demonstrate that our ALD can properly obtain samples from target distributions in conditional and unconditional cases. First, we use examples where the posterior density can be derived in a closed-form. We initially generate three synthetic data  $x_{1}, x_{2}, x_{3}$ , where each  $x_{i}$  is sampled from a bivariate Gaussian distribution as follows:

$$
p \left(\boldsymbol {z}\right) = \mathcal {N} \left(\boldsymbol {z}; \boldsymbol {\mu_ {z}}, \boldsymbol {\Sigma_ {z}}\right), \quad p \left(\boldsymbol {x} \mid \boldsymbol {z}\right) = \mathcal {N} \left(\boldsymbol {x}; \boldsymbol {z}, \boldsymbol {\Sigma_ {x}}\right).
$$

In this case, we can calculate the exact posterior as follows:

$$
p \left(\boldsymbol {z} \mid \boldsymbol {x}\right) = \mathcal {N} \left(\boldsymbol {z}; \left(\boldsymbol {\Sigma_ {z}} ^ {- 1} + \boldsymbol {\Sigma_ {x}} ^ {- 1}\right) ^ {- 1} \left(\boldsymbol {\Sigma_ {z}} ^ {- 1} \boldsymbol {\mu_ {z}} + \boldsymbol {\Sigma_ {x}} ^ {- 1} \boldsymbol {x}\right), \left(\boldsymbol {\Sigma_ {z}} ^ {- 1} + \boldsymbol {\Sigma_ {x}} ^ {- 1}\right) ^ {- 1}\right),
$$

![](images/c079531a50b673553527ac7fcc04f8735e38e8068a3cc21b811b366784b575de.jpg)  
(A) Conditional Case

![](images/a1f98d4aa69f81404080df6870f549f5148035561c347d214e58770afcbb4368.jpg)  
Figure 2: Visualization of ground truth density (left) and samples by ALD (right) in the conditional case (A) and the unconditional case (B) in toy examples.

![](images/90801aae9e606ea8a7fd60c36c010f09e8cea3b3112843233a97accaa4203e79.jpg)  
(B) Unconditional Case

![](images/ae35f3e431fe2959ba989f53f2b070a2182fa0cd8ab0a291ccf655255e601946.jpg)

![](images/59f462fb715dbf21142d0d108666a5ff7392232c417067f465dd187786426826.jpg)  
Figure 3: Evolution of sample values across MCMC iterations for traditional LD and our SGALD in univariate Gaussian examples. The black lines denote the ground truth posteriors (the solid lines show the mean values, and the dashed lines show the standard deviation).

![](images/5c48834088c735bec1eee5b9ddd47f71d969883b6d3dd5396e16191ed689f3f1.jpg)

![](images/f329e0bca1d76f0c5c5c56758a685d6165fe513234a103f926eb130910ae93a8.jpg)

In this experiment, we set  $\pmb{\mu_{\mathbf{z}}} = \left[ \begin{array}{l}0\\ 0 \end{array} \right],\pmb{\Sigma_{\mathbf{z}}} = \left[ \begin{array}{ll}1 & 0\\ 0 & 1 \end{array} \right]$ , and  $\pmb{\Sigma_{\mathbf{x}}} = \left[ \begin{array}{ll}0.7 & 0.6\\ 0.7 & 0.8 \end{array} \right]$ . We simulate our ALD algorithm for this setting to obtain samples from the posterior. We use a neural network of three fully connected layers of 128 units with ReLU activation for the inference model  $f_{\mathbf{z}|\mathbf{x}}$ ; setting the step size to  $4\times 10^{-4}$ , and update the parameters for 3,000 steps. We omit the first 1,000 samples as burn-in steps and use the remaining 2,000 samples for qualitative evaluation. The result is shown in Figure 2 (A). ALD produces samples that match the shape of the target distributions well, even though ALD does not perform direct updates of samples in the latent space. We also performed a similar experiment in a univariate setting to see the convergence speed of our SGALD, the minibatch version of ALD (see Appendix D.1 for the detailed experimental setting). Figure 3 shows the evolution of obtained sample values by traditional LD and our SGALD. It can be observed that SGALD's samples converge much faster than traditional LD.

In addition to the simple conjugate Gaussian example, we experiment with a complex posterior, wherein the likelihood is defined with a randomly initialized neural network. For comparison, we also implement the amortized variational inference (AVI) method, in which the posterior is approximated with a Gaussian distribution parameterized by a neural network (see Appendix D.2 for more experimental details). Figure 4 shows a typical example, which characterizes the difference between AVI and ALD. The advantage of our ALD over AVI is the flexibility of posterior approximation. AVI methods typically approximate posteriors using variational distribu

tions, which have tractable density functions. Hence, their approximation power is limited by the choice of variational distribution family, and they often fail to approximate such complex posteriors. On the other hand, ALD can capture such posteriors well. The results in other examples are summarized in Figure 5 in the appendix.

Furthermore, we also test our ALD for sampling from an unconditional distribution. In this experiment, we use a mixture distribution of eight Gaussians and obtain samples using ALD, as shown in

![](images/d5255300801718660681761cdb55f1129240e6bffa6028b45a478a9115ffa07c.jpg)  
GT

![](images/5b7ffd65181dab776328a8a814e41355b8c51be8965bc628844816a76329a921.jpg)  
AVI

![](images/46303a9f4fd85ecc642ad08ad59448386038ac8a8cd6250b0c62ced5caae9ae3.jpg)  
Figure 4: Visualizations of a ground truth posterior (left), an approximation by AVI (center), and samples by ALD (right) in the neural likelihood example.  
ALD

Table 1: Quantitative results of the image generation for SVHN, CIFAR-10, and CelebA-HQ. We report the mean and standard deviation of the Fréchet Inception Distance in three different seeds.  

<table><tr><td rowspan="2"></td><td rowspan="2">Description</td><td>SVHN</td><td>CIFAR-10</td><td colspan="2">CelebA-HQ</td></tr><tr><td>32 × 32</td><td>32 × 32</td><td>32 × 32</td><td>64 × 64</td></tr><tr><td>VAE</td><td>VI + amortization</td><td>59.07 ± 0.79</td><td>108.8 ± 0.2</td><td>114.3 ± 2.6</td><td>189.8 ± 1.0</td></tr><tr><td>VAE-flow</td><td>VI + amortization + flow</td><td>60.03 ± 0.92</td><td>114.5 ± 1.7</td><td>132.6 ± 6.1</td><td>194.4 ± 4.2</td></tr><tr><td>ABP</td><td>LD</td><td>98.63 ± 10.62</td><td>162.8 ± 3.6</td><td>99.27 ± 2.42</td><td>145.2 ± 8.1</td></tr><tr><td>DLGM</td><td>LD + amortized init.</td><td>63.54 ± 0.76</td><td>115.2 ± 1.1</td><td>152.2 ± 2.53</td><td>222.3 ± 1.4</td></tr><tr><td>LEBM</td><td>LD + EBM</td><td>115.4 ± 3.6</td><td>181.9 ± 4.3</td><td>62.36 ± 8.40</td><td>78.80 ± 1.24</td></tr><tr><td>LAE</td><td>LD + EBM + amortization</td><td>46.66 ± 1.33</td><td>95.85 ± 1.06</td><td>40.33 ± 1.33</td><td>61.38 ± 1.20</td></tr></table>

Figure 2 (B). We can observe that ALD adequately captures the actual density's multimodality and works well in the unconditional case.

# 6.2 IMAGE GENERATION

To demonstrate the applicability of our LAE to the generative model training, we experiment on image generation tasks using SVHN, CIFAR10, and CelebA-HQ datasets. Note that our goal here is not to provide the state-of-the-art results on image generation benchmarks but to verify the effectiveness of our ALD as a method of approximate inference in deep latent variable models. For this aim, we compare our LAE with five baseline methods, as shown in Table 1. VAE (Kingma & Welling, 2013) is one of the most popular deep latent variable models in which the posterior distribution is approximated using the AVI. VAE-flow is an extension of VAE in which the flexibility of AVI is improved using normalizing flows. In addition to AVI-based methods, we use three methods based on Langevin dynamics (LD). The alternating back-propagation (ABP) uses traditional LD to approximate the posterior, and the deep latent Gaussian model (DLGM) uses a VAE-like inference model to initialize LD. The latent energy-based model (LEBM) uses an EBM for the latent prior, and the EBM and posterior sampling is performed via traditional LD. LEBM can be regarded as a non-amortization version of our LAE.

We apply a commonly used convolutional neural network-based architecture for all models and a multi-layer perceptron for an energy-based model in the latent space of LAE and LEBM. Please refer to Appendix D.3 for more detailed experimental settings. For quantitative evaluation of the sample quality, we report the Fréchet Inception Distance (FID) (Heusel et al., 2017).

The results are summarized in Table 1. It can be observed that LAE consistently outperforms the baseline models in terms of FID. In training speed, LAE takes 34.12 seconds per epoch on average to train with CIFAR-10, while LEBM and DLGM take 157.6 and 45.75 seconds per epoch, respectively. This result shows that LAE is approximately four times faster than the non-amortized LEBM and 1.34 times faster than the partially amortized DLGM.

# 7 CONCLUSION

This paper proposed amortized Langevin dynamics (ALD), an efficient MCMC method for deep latent variable models. The ALD amortizes the cost of datapoint-wise iterations by using inference models. We showed that our ALD algorithm could accurately approximate posteriors with both theoretical and empirical studies. Using ALD, we derived a novel scheme of deep generative models called the Langevin autoencoder (LAE). We demonstrated that our LAE performs better than existing deep latent variable models in sample quality and training speed.

This study will be the first step to further work on efficient MCMC for latent variable models with large-scale datasets. For instance, deriving a Metropolis-Hastings rejection step for ALD and algorithms based on more sophisticated Hamiltonian Monte Carlo methods is an exciting direction of future work.

# REFERENCES

Jinwon An and Sungzoon Cho. Variational autoencoder based anomaly detection using reconstruction probability. *Special Lecture on IE*, 2(1), 2015.  
Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan. arXiv preprint arXiv:1701.07875, 2017.  
Chris Cremer, Xuechen Li, and David Duvenaud. Inference suboptimality in variational autoencoders. In International Conference on Machine Learning, pp. 1078-1086. PMLR, 2018.  
Kingma DP and Jimmy Ba. Adam: A method for stochastic optimization. In Proc. of the 3rd International Conference for Learning Representations (ICLR), 2015.  
SM Ali Eslami, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S Morcos, Marta Garnelo, Avraham Ruderman, Andrei A Rusu, Ivo Danihelka, Karol Gregor, et al. Neural scene representation and rendering. Science, 360(6394):1204-1210, 2018.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.  
Tian Han, Yang Lu, Song-Chun Zhu, and Ying Nian Wu. Alternating back-propagation for generator network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.  
Hao He, Hao Wang, Guang-He Lee, and Yonglong Tian. Probgan: Towards probabilistic gan with theoretical guarantees. In International Conference on Learning Representations, 2018.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.  
Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504-507, 2006.  
Matthew D Hoffman. Learning deep latent gaussian models with markov chain monte carlo. In International conference on machine learning, pp. 1510-1519. PMLR, 2017.  
Chin-Wei Huang, David Krueger, Alexandre Lacoste, and Aaron Courville. Neural autoregressive flows. In International Conference on Machine Learning, pp. 2078-2087. PMLR, 2018.  
Rie Johnson and Tong Zhang. Composite functional gradient learning of generative adversarial models. In International Conference on Machine Learning, pp. 2371-2379. PMLR, 2018.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in neural information processing systems, pp. 3581-3589, 2014.  
Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Advances in neural information processing systems, pp. 4743-4751, 2016.  
Peter E Kloeden and Eckhard Platen. Numerical solution of stochastic differential equations, volume 23. Springer Science & Business Media, 2013.  
Ananya Kumar, SM Eslami, Danilo J Rezende, Marta Garnelo, Fabio Viola, Edward Lockhart, and Murray Shanahan. Consistent generative query networks. arXiv preprint arXiv:1807.02033, 2018.  
Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin. Preconditioned stochastic gradient Langevin dynamics for deep neural networks. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.

Yingzhen Li, Richard E Turner, and Qiang Liu. Approximate inference with amortised mcmc. arXiv preprint arXiv:1702.08343, 2017.  
Radford M Neal. Mcmc using hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, pp. 113, 2011.  
Andrew Ng et al. Sparse autoencoder. CS294A Lecture notes, 72(2011):1-19, 2011.  
Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. Learning latent space energy-based prior model. arXiv preprint arXiv:2006.08205, 2020a.  
Bo Pang, Erik Nijkamp, Jiali Cui, Tian Han, and Ying Nian Wu. Semi-supervised learning by latent space energy-based model of symbol-vector coupling. arXiv preprint arXiv:2010.09359, 2020b.  
Prajit Ramachandran, Barret Zoph, and Quoc V Le. Swish: a self-gated activation function. arXiv preprint arXiv:1710.05941, 7:1, 2017.  
Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pp. 1530-1538. PMLR, 2015.  
Yunus Saatci and Andrew Wilson. Bayesian gans. In Advances in neural information processing systems, pp. 3624-3633, 2017.  
Rui Shu, Hung H Bui, Shengjia Zhao, Mykel J Kochenderfer, and Stefano Ermon. Amortized inference regularization. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 4398-4407, 2018.  
Jinsong Su, Shan Wu, Deyi Xiong, Yaojie Lu, Xianpei Han, and Biao Zhang. Variational recurrent neural machine translation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
Michalis K Titsias and Francisco Ruiz. Unbiased implicit variational inference. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 167-176. PMLR, 2019.  
Rianne Van Den Berg, Leonard Hasenclever, Jakub M Tomczak, and Max Welling. Sylvester normalizing flows for variational inference. In 34th Conference on Uncertainty in Artificial Intelligence 2018, UAI 2018, pp. 393-402. Association For Uncertainty in Artificial Intelligence (AUAI), 2018.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681-688. CiteSeer, 2011.  
Jianwen Xie, Ruiqi Gao, Zilong Zheng, Song-Chun Zhu, and Ying Nian Wu. Learning dynamic generator model by alternating back-propagation through time. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5498-5507, 2019.  
Xianglei Xing, Ruiqi Gao, Tian Han, Song-Chun Zhu, and Ying Nian Wu. Deformable generator network: Unsupervised disentanglement of appearance and geometry. arXiv preprint arXiv:1806.06298, 2018.  
Jing Zhang, Jianwen Xie, and Nick Barnes. Learning noise-aware encoder-decoder from noisy labels by alternating back-propagation for saliency detection. arXiv preprint arXiv:2007.12211, 2020.  
Yizhe Zhu, Jianwen Xie, Bingchen Liu, and Ahmed Elgammal. Learning feature-to-feature translator by alternating back-propagation for generative zero-shot learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9844-9854, 2019.
