# LAGGING INFERENCE NETWORKS AND POSTERIOR COLLAPSE IN VARIATIONAL AUTOENCODERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The variational autoencoder (VAE) is a popular combination of deep latent variable model and accompanying variational learning technique. By using a neural inference network to approximate the model's posterior on latent variables, VAEs efficiently parameterize a lower bound on marginal data likelihood that can be optimized directly via gradient methods. In practice, however, VAE training often results in a degenerate local optimum known as "posterior collapse" where the model learns to ignore the latent variable and the approximate posterior mimics the prior. In this paper, we investigate posterior collapse from the perspective of training dynamics. We find that during the initial stages of training the inference network fails to approximate to the model's true posterior, which is a moving target. As a result, the model is encouraged to ignore the latent encoding and posterior collapse results. Based on this observation, we propose an extremely simple modification to VAE training to reduce inference lag: depending on the model's current mutual information between latent variable and observation, we aggressively optimize the inference network before performing each model update. Despite introducing neither new model components nor significant complexity over basic VAE, our approach is able to avoid the problem of collapse that has plagued a large amount of previous work. Empirically, our approach outperforms strong autoregressive baselines on text and image benchmarks in terms of held-out likelihood, and is competitive with more complex techniques for avoiding collapse, while being substantially faster.

# 1 INTRODUCTION

Variational autoencoders (VAEs) (Kingma & Welling, 2014) represent a popular combination of a deep latent variable model (shown in Figure 1(a)) and an accompanying variational learning technique. The generative model in VAE defines a marginal distribution on observations,  $\mathbf{x} \in \mathcal{X}$ , as:

$$
p _ {\boldsymbol {\theta}} (\mathbf {x}) = \int p _ {\boldsymbol {\theta}} (\mathbf {x} | \mathbf {z}) p (\mathbf {z}) \mathrm {d} \mathbf {z}. \tag {1}
$$

The model's generator defines  $p_{\theta}(\mathbf{x}|\mathbf{z})$  and is typically parameterized as a complex neural network. Standard training involves optimizing an evidence lower bound (ELBO) on the intractable marginal data likelihood (Eq.1), where an auxiliary variational distribution  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is introduced to approximate the model posterior distribution  $p_{\theta}(\mathbf{z}|\mathbf{x})$ . VAEs make this learning procedure highly scalable to large datasets by sharing parameters in the inference network to amortize the cost. This amortization is in contrast to traditional variational inference that has separate local variational parameters for every data point (Hoffman et al., 2013).

While successful on some datasets, prior works has found that VAE training often suffers from "posterior collapse", in which the model ignores the latent variable  $\mathbf{z}$  when the generator  $p_{\theta}(\mathbf{x}|\mathbf{z})$  is parametrized using a strong autoregressive neural network such as LSTMs for text and PixelCNNs (van den Oord et al., 2016) for images (Bowman et al., 2016; Chen et al., 2017; Kingma et al., 2016; Yang et al., 2017; Dieng et al., 2018). Posterior collapse is especially evident when modeling discrete data, which hinders the usage of VAEs in important applications like natural language processing. Existing work analyze this problem from a static optimization perspective noting that the collapsed solution is often a reasonably good local optimum in terms of ELBO (Chen et al., 2017;

![](images/25432c0d8c36fd7b6c80ddbee2e985463ff41bfa7ae23f5fdfcd01f82b2f37f0.jpg)  
(a) Variational autoencoders

![](images/35bc0f18508bb3af865f0a8989ec798b6982ebd77a72f86e90c974657a75b8c5.jpg)  
(b) Posterior mean space  
Figure 1: Left: Depiction of generative model  $p(\mathbf{z})p_{\theta}(\mathbf{x}|\mathbf{z})$  and inference network  $q_{\phi}(\mathbf{z}|\mathbf{x})$  in VAEs. Right: A toy posterior mean space  $(\mu_{\mathbf{x},\theta},\mu_{\mathbf{x},\phi})$  with scalar  $z$ . The horizontal axis represents the mean of the model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$ , and the vertical axis represents the mean of the approximate posterior  $q_{\phi}(\mathbf{z}|\mathbf{x})$ . The dashed diagonal line represents when the approximate posterior matches the true model posterior in terms of mean.

Zhao et al., 2017; Alemi et al., 2018). Thus, many proposed solutions to posterior collapse focus on weakening the generator by replacing it with a non-recurrent alternative (Yang et al., 2017; Semeniuta et al., 2017) or modifying the training objective (Zhao et al., 2017; Tolstikhin et al., 2018). In this paper, we try to analyze the problem from perspective of training dynamics, and propose a novel training procedure for VAEs that optimizes the standard ELBO objective without changing the model structure.

Recently Kim et al. (2018) proposed a new approach to training VAEs by composing the standard inference network with additional mean-field updates. The resulting semi-amortized approach empirically avoided collapse and obtained better ELBO. However, because of the costly instance-specific local inference steps, the new method is more than 10x slower than basic VAE training in practice. It is also unclear why the basic VAE method fails to find better local optima that make use of latents. We consider two questions in this paper: (1) Why does basic VAE training often fall into undesirable collapsed local optima? (2) Is there a simpler way to change the training trajectory to find a better non-trivial local optimum?

To this end, we first study the posterior collapse problem from the perspective of training dynamics, and empirically find that the approximate posterior is lagging far behind the true model posterior in the initial stages of training (Section 3). We then demonstrate empirically how such lagging behavior can drive the generative model towards a collapsed local optimum, and propose a novel training procedure for VAEs that aggressively optimizes the inference network with more updates to mitigate the lagging issue (Section 4). Without introducing new modeling components over basic VAEs or additional complexity, our approach is surprisingly simple yet effective in circumventing posterior collapse. As a density estimator it outperforms neural autoregressive baselines on both text (Yahoo and Yelp) and image (OMNIGLOT) benchmarks, leading to comparable performance with more complicated previous state-of-the-art methods at a fraction of the training cost (Section 6).<sup>1</sup>

# 2 BACKGROUND

# 2.1 VARIATIONAL AUTOENCODERS

VAEs learn deep generative models defined by a prior  $p(\mathbf{z})$  and a conditional distribution  $p_{\theta}(\mathbf{x}|\mathbf{z})$  as shown in Figure 1(a). In most cases the marginal data likelihood is intractable, so VAEs (Kingma & Welling, 2014) instead optimize a tractable variational lower bound (ELBO) of  $\log p_{\theta}(\mathbf{x})$ ,

$$
\mathcal {L} (\mathbf {x}; \theta , \phi) = \underbrace {\mathbb {E} _ {\mathbf {z} \sim q _ {\phi} (\mathbf {z} | \mathbf {x})} [ \log p _ {\theta} (\mathbf {x} | \mathbf {z}) ]} _ {\text {R e c o n s t r u c t i o n L o s s}} - \underbrace {D _ {\mathrm {K L}} \left(q _ {\phi} (\mathbf {z} | \mathbf {x}) \| p (\mathbf {z})\right)} _ {\text {K L R e g u l a r i z e r}}, \tag {2}
$$

where  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is a variational distribution parameterized by an inference network with parameters  $\phi$ , and  $p_{\theta}(\mathbf{x}|\mathbf{z})$  denotes the generator network with parameters  $\theta$ .  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is optimized to approximate the model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$ . This lower bound is composed of a reconstruction loss term that encourages the inference network to encode information necessary to generate the data and a KL regularizer to push  $q_{\phi}(\mathbf{z}|\mathbf{x})$  towards the prior  $p(\mathbf{z})$ . Below, we consider  $p(\mathbf{z}) \coloneqq \mathcal{N}(\mathbf{0},\mathbf{I})$  unless otherwise specified. A key advantage of using inference networks (also called amortized inference) to train deep generative models over traditional locally stochastic variational inference (Hoffman et al., 2013) is that they share parameters over all data samples, amortizing the computational cost and allowing for efficient training.

The term VAE is often used both to denote the class of generative models and the amortized inference procedure used in training. In this paper, it is important to distinguish the two and throughout we will refer to the generative model as the VAE-model, and the training procedure as VAE-training.

# 2.2 POSTERIOR COLLAPSE

Despite VAE's appeal as a tool to learn unsupervised representations through the use of latent variables, as mentioned VAE-models are often found to ignore latent variable when using a flexible generator like in the introduction LSTMs (Bowman et al., 2016). This problem of "posterior collapse", occurs when the training procedure falls into the trivial local optimum of the ELBO objective in which both the variational posterior and true model posterior collapse to the prior. This is undesirable because an important goal of VAEs is to learn meaningful latent features for inputs. Mathematically, posterior collapse represents a local optimum of VAEs where  $q_{\phi}(\mathbf{z}|\mathbf{x}) = p_{\theta}(\mathbf{z}|\mathbf{x}) = p(\mathbf{z})$  for all  $\mathbf{x}$ . To facilitate our analysis about the cause leading up to collapse, we further define two partial collapse states for each  $\mathbf{x}$ : model collapse, when  $p_{\theta}(\mathbf{z}|\mathbf{x}) = p(\mathbf{z})$ , and inference collapse, when  $q_{\phi}(\mathbf{z}|\mathbf{x}) = p(\mathbf{z})$ . Note that in this paper we use these two terms to denote the posterior states in the middle of training instead of local optima at the end. These two partial collapse states may not happen at the same time, which we will discuss later.

# 2.3 VISUALIZATION OF POSTERIOR DISTRIBUTION

Posterior collapse is closely related to true model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$  and approximate posterior  $q_{\phi}(\mathbf{z}|\mathbf{x})$  as it is defined. Thus, in order to observe how posterior collapse happens, we track the state of  $p_{\theta}(\mathbf{z}|\mathbf{x})$  and  $q_{\phi}(\mathbf{z}|\mathbf{x})$  over the course of training, we will analyze the training trajectory in terms of the posterior mean space  $\mathcal{U} = \{\mu : \mu = (\mu_{\mathbf{x},\theta}^T, \mu_{\mathbf{x},\phi}^T)\}$ , where  $\mu_{\mathbf{x},\theta}$  and  $\mu_{\mathbf{x},\phi}$  are the means of  $p_{\theta}(\mathbf{z}|\mathbf{x})^2$  and  $q_{\phi}(\mathbf{z}|\mathbf{x})$ , respectively. We can then roughly consider  $\mu_{\mathbf{x},\theta} = \mathbf{0}$  as model collapse and  $\mu_{\mathbf{x},\phi} = \mathbf{0}$  as inference collapse as we defined before. Each  $\mathbf{x}$  will be projected to a point in this space under the current model and inference network parameters. If  $\mathbf{z}$  is a scalar we can efficiently compute  $\mu_{\mathbf{x},\theta}$  and visualize the posterior mean space as shown in Figure 1(b). The diagonal line  $\mu_{\mathbf{x},\theta} = \mu_{\mathbf{x},\phi}$  represents parameter settings where  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is equal to  $p_{\theta}(\mathbf{z}|\mathbf{x})$  in terms of mean, indicating a well-trained inference network. The posterior collapsed local optimum is located at the origin, while more desirable local optima may be located on the diagonal. In this paper we will utilize this posterior mean space multiple times to analyze the posterior dynamics.

# 3 A LAGGING INFERENCE NETWORK PREVENTS USING LATENT CODES

In this section we analyze posterior collapse from a perspective of training dynamics. We will answer the question of why the basic VAE-training with strong decoders tends to hit a collapsed local optimum and provide intuition for the simple solution we propose in Section 4.

# 3.1 INTUITIONS FROM ELBO

Since posterior collapse is directly relevant to the approximate posterior  $q_{\phi}(\mathbf{z}|\mathbf{x})$  and true model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$ , we aim to analyze their training dynamics to study how posterior collapse happens.

![](images/79deb5d1a9334ef705ec54dfdd267fddd613e0aeab5d6eac4d832e354cb77453.jpg)

![](images/5bca95b6c5bd31c7296ec24470a281d0122ea7a0477c6d211cc8d79dbf15af7c.jpg)

![](images/47fdbac5235fc65e54f9ab3e700e7d42dc16513b852895d40a793ad5f54ec246.jpg)

![](images/d9a8be05612bad5a90261caac2675402c12e407509c3f026ba0f0ccc9196f123.jpg)

![](images/32bb2c8b164678472735d470ffad02414bd7ccae57ce7027e546f01e2a81a103.jpg)  
Figure 2: The projections of 500 data samples from synthetic dataset on the posterior mean space over the course of training. The top row is from the basic VAE training, the bottom row is from our aggressive inference network training. The results show that while the approximate posterior is lagging far behind the true model posterior in basic VAE-training, our aggressive training approach successfully moves the points onto the diagonal line and away from inference collapse.

![](images/c48ba3ecdea4c50ba167215e4f656cf48ef086cffe4987c89f072728e8c47250.jpg)

![](images/f06b0209cf15b7bd8e65bc12c9bd985a1813201c183dd4f32db9d65f26fa5910.jpg)

![](images/f0cab0612bc3facd719b4c9ad1c56a778dfb8e4dcc0b9120fc3274ebbe7f22a1.jpg)

To this end, it is useful to analyze an alternate form of ELBO:

$$
\mathcal {L} (\mathbf {x}; \boldsymbol {\theta}, \phi) = \underbrace {\log p _ {\boldsymbol {\theta}} (\mathbf {x})} _ {\text {m a r g i n a l} \log \text {d a t a l i k e l i h o o d}} - \underbrace {D _ {\mathrm {K L}} \left(q _ {\phi} (\mathbf {z} | \mathbf {x}) \mid p _ {\boldsymbol {\theta}} (\mathbf {z} | \mathbf {x})\right)} _ {\text {a g r e e m e n t b e t w e e n a p p r o x i m a t e a n d m o d e l p o s t e r i o r s}}, \tag {3}
$$

With this view, the only goal of approximate posterior  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is to match model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$ , the optimization of  $p_{\theta}(\mathbf{z}|\mathbf{x})$  is influenced by two forces, one of which is the ideal objective marginal data likelihood, and the other is  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x})|p_{\theta}(\mathbf{z}|\mathbf{x}))$ , which drives  $p_{\theta}(\mathbf{z}|\mathbf{x})$  towards  $q_{\phi}(\mathbf{z}|\mathbf{x})$ . Ideally if the approximate posterior is perfect, the second force will vanish, with  $\nabla_{\theta}D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x})|p_{\theta}(\mathbf{z}|\mathbf{x})) = 0$  when  $q_{\phi}(\mathbf{z}|\mathbf{x}) = p_{\theta}(\mathbf{z}|\mathbf{x})$ . At the start of training,  $\mathbf{z}$  and  $\mathbf{x}$  are nearly independent under both  $q_{\phi}(\mathbf{z}|\mathbf{x})$  and  $p_{\theta}(\mathbf{z}|\mathbf{x})$  as we show in Section 3.2, i.e. all  $\mathbf{x}$  suffer from both model collapse and inference collapse in the beginning. Then the only component in the training objective that possibly causes dependence between  $\mathbf{z}$  and  $\mathbf{x}$  under  $p_{\theta}(\mathbf{z}|\mathbf{x})$  is  $\log p_{\theta}(\mathbf{x})$ . However, this pressure may be overwhelmed by the KL term when  $p_{\theta}(\mathbf{z}|\mathbf{x})$  and  $q_{\phi}(\mathbf{z}|\mathbf{x})$  starts to diverge. We hypothesize that, in practice, training drives  $p_{\theta}(\mathbf{z}|\mathbf{x})$  and  $q_{\phi}(\mathbf{z}|\mathbf{x})$  to the prior in order to bring them into alignment, while locking into model parameters that capture the distribution of  $\mathbf{x}$  while ignoring  $\mathbf{z}$ . Critically, posterior collapse is a local optimum; once a set of parameters that achieves these goals are reached, gradient optimization fails to make further progress, even if better overall models that make use of  $\mathbf{z}$  to describe  $\mathbf{x}$  exist.

Next we visualize the posterior mean space by training a basic VAE with a scalar latent variable on a relatively simple synthetic dataset to examine our hypothesis.

# 3.2 OBSERVATIONS ON SYNTHETIC DATA

As a synthetic dataset we use discrete sequence data, as posterior collapse has been found the most severe in text modeling tasks. Details on this synthetic dataset and experiment are in Appendix B.1.

We train a basic VAE with a scalar latent variable, LSTM encoder, and LSTM decoder on our synthetic dataset. We sample 500 data points from validation set and show them on the posterior mean space plots at four different training stages from initialization to convergence in Figure 2. The mean of the approximate posterior distribution  $\mu_{\mathbf{x},\phi}$  is from the output of the inference network, and  $\mu_{\mathbf{x},\theta}$  can be approximated by discretization of true model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$  (see Appendix A).

As illustrated in Figure 2, all points are located at the origin upon initialization, which means  $\mathbf{z}$  and  $\mathbf{x}$  are both almost independent and zero mean at the beginning of training. In the second stage of basic VAE, the points start to spread along the  $\mu_{\mathbf{x},\theta}$  axis. This phenomenon implies that for some data points  $p_{\theta}(\mathbf{z}|\mathbf{x})$  moves far away from the prior  $p(\mathbf{z})$ , and confirms that  $\log p_{\theta}(\mathbf{x})$  is able to help

Algorithm 1 VAE training with controlled aggressive inference network optimization.  
1:  $\theta, \phi \gets$  Initialize parameters  
2: aggressive  $\leftarrow$  TRUE  
3: repeat  
4: if aggressive then  
5: repeat ▷ [aggressive updates]  
6: X  $\leftarrow$  Random data minibatch  
7: Compute gradients  $g \gets \nabla_{\phi} \mathcal{L}(\mathbf{X}; \theta, \phi)$   
8: Update  $\phi$  using gradients  $g$   
9: until convergence  
10: X  $\leftarrow$  Random data minibatch  
11: Compute gradients  $g \gets \nabla_{\theta} \mathcal{L}(\mathbf{X}; \theta, \phi)$   
12: Update  $\phi$  using gradients  $g$   
13: else ▷ [basic VAE training]  
14: X  $\leftarrow$  Random data minibatch  
15: Compute gradients  $g \gets \nabla_{\phi, \theta} \mathcal{L}(\mathbf{X}; \theta, \phi)$   
16: Update  $\theta, \phi$  using  $g$   
17: end if  
18: Update aggressive as discussed in Section 4.2  
19: until convergence

![](images/b4bc18c0bc299a4c83c5e268e2ee9da1f344603a1662997bf3f8f0e4e3357ee8.jpg)  
Figure 3: Trajectory of one data instance on the posterior mean space with our aggressive training procedure. Horizontal arrow denotes one step of generator update, and vertical arrow denotes the inner loop of inference network update. We note that the approximate posterior  $q_{\phi}(\mathbf{z}|\mathbf{x})$  takes an aggressive step to catch up to the model posterior  $p_{\theta}(\mathbf{z}|\mathbf{x})$ .

move away from model collapse. However, all of these points are still close to the  $\mu_{\mathbf{x},\phi}$  axis, which suggests that  $q_{\phi}(\mathbf{z}|\mathbf{x})$  fails to catch up to  $p_{\theta}(\mathbf{z}|\mathbf{x})$  and these points are still in a state of inference collapse. As expected, the dependence between  $\mathbf{z}$  and  $\mathbf{x}$  under  $p_{\theta}(\mathbf{z}|\mathbf{x})$  is gradually lost and finally the model converges to the collapsed local optimum.

# 4 METHOD

# 4.1 AGGRESSIVE TRAINING OF THE INFERENCE NETWORK

The problem reflected in Figure 2 implies that the inference network is lagging far behind  $p_{\theta}(\mathbf{z}|\mathbf{x})$ , and might suggest more "aggressive" inference network updates are needed. Instead of blaming the poor approximation on the limitation of the inference network's amortization, we hypothesize that the optimization of the inference and generation networks are imbalanced, and propose to separate the optimization of the two. Specifically, we change the training procedure to:

$$
\boldsymbol {\theta} ^ {*} = \underset {\boldsymbol {\theta}} {\arg \max } \mathcal {L} (\mathbf {X}; \boldsymbol {\theta}, \boldsymbol {\phi} ^ {*}), \text {w h e r e} \boldsymbol {\phi} ^ {*} = \underset {\boldsymbol {\phi}} {\arg \max } \mathcal {L} (\mathbf {X}; \boldsymbol {\theta}, \boldsymbol {\phi}), \tag {4}
$$

where optimizing the inference network  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is an inner loop in the entire training process as shown in Algorithm 1. This training procedure shares the same spirit with traditional stochastic variational inference (SVI) (Hoffman et al., 2013) that performs iterative inference for each data point separately and suffers from very lengthy iterative estimation. Compared with recent work that try to combine amortized variational inference and SVI (Hjelm et al., 2016; Krishnan et al., 2018; Kim et al., 2018; Marino et al., 2018) where the inference network is learned to be a component to help perform instance-specific variational inference, our approach keeps variational inference fully amortized, allowing for reverting back to efficient basic VAE training as discussed in Section 4.2. Also, this aggressive inference network optimization algorithm is as simple as basic VAE training without introducing additional SVI steps, yet attains comparable performance to more sophisticated approaches as we will show in Section 6.

# 4.2 STOPPING CRITERION

Always training with Eq.4 would be inefficient and neglects the benefit of the amortized inference network. Following our previous analysis, the term  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x})|p_{\theta}(\mathbf{z}|\mathbf{x}))$  tends to pressure  $q_{\phi}(\mathbf{z}|\mathbf{x})$  or  $p_{\theta}(\mathbf{z}|\mathbf{x})$  to  $p(\mathbf{z})$  only if at least one of them is close to  $p(\mathbf{z})$ , and thus we posit that if we can confirm that we haven't reached this degenerate condition, we can continue with standard VAE training. Since  $q_{\phi}(\mathbf{z}|\mathbf{x})$  is the one lagging behind, we use the mutual information  $I_{q}$  between  $\mathbf{z}$  and

$\mathbf{x}$  under  $q_{\phi}(\mathbf{z}|\mathbf{x})$  to control our stopping criterion. In practice, we compute the mutual information on the validation set every epoch, and stop the aggressive updates when  $I_{q}$  stops climbing. In all our experiments in this paper we found that the aggressive algorithm usually reverts back to basic VAE-training within 5 epochs. Mutual information,  $I_{q}$  is computed by:

$$
I _ {q} = \mathbb {E} _ {\mathbf {x} \sim p _ {d} (\mathbf {x})} \left[ D _ {\mathrm {K L}} \left(q _ {\phi} (\mathbf {z} | \mathbf {x}) \| p (\mathbf {z})\right) \right] - D _ {\mathrm {K L}} \left(q _ {\phi} (\mathbf {z}) \| p (\mathbf {z})\right), \tag {5}
$$

where  $p_d(\mathbf{x})$  is the empirical distribution. The aggregated posterior,  $q_{\phi}(\mathbf{z}) = \mathbb{E}_{\mathbf{x} \sim p_d(\mathbf{x})}[q_{\phi}(\mathbf{z}|\mathbf{x})]$ , can be approximated with a Monte Carlo estimate.  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z})||p(\mathbf{z}))$  is also approximated by Monte Carlo, where samples from  $q_{\phi}(\mathbf{z})$  can be easily obtained by ancestral sampling (i.e. sample  $\mathbf{x}$  from dataset and sample  $\mathbf{z} \sim q_{\phi}(\mathbf{z}|\mathbf{x})$ ). The complete algorithm is shown in Algorithm 1.

# 4.3 OBSERVATIONS ON SYNTHETIC DATASET

By training the VAE model with our approach on synthetic data, we visualize the 500 data samples in the posterior mean space in Figure 2. From this, it is evident that the points move towards  $\mu_{\mathbf{x},\theta} = \mu_{\mathbf{x},\phi}$  and are roughly distributed along the diagonal in the end. This is in striking contrast to the basic VAE and confirms our hypothesis that the inference and generator optimization can be rebalanced by simply performing more updates of the inference network. In Figure 3 we show the training trajectory of one single data instance for the first several optimization iterations and observe how the aggressive updates help escape inference collapse.

# 5 RELATION TO RELATED WORK

Posterior collapse in VAEs is first detailed in (Bowman et al., 2016) where they combine a LSTM decoder with VAE for text modeling. They interpret this problem from a regularization perspective, and propose the "KL cost annealing" method to address this issue, whereby the weight of KL term between approximate posterior and prior increases from a small value to one in a "warm-up" period. This method has been shown to be unable to deal with collapse on complex text datasets with very large LSTM decoders (Yang et al., 2017; Kim et al., 2018). Many works follow this line to lessen the effect of KL term such as  $\beta$ -VAE (Higgins et al., 2017) that treats the KL weight as a hyperparameter or "free bits" method that constrains the minimum value of the KL term. Our approach differs from these methods in that we do not change ELBO objective during training and in principle maximum likelihood estimation. While these methods explicitly encourages the use of latent variable, they may implicitly sacrifice density estimation performance at the same time, as we will discuss in Section 6.

Another thread of research focuses on a different problem called the "amortization gap" (Cremer et al., 2018) that refers to the difference of ELBO caused by parameter sharing of the inference network. Some approaches try to combine traditional instance-specific variational inference with amortized variational inference to narrow this gap (Hjelm et al., 2016; Krishnan et al., 2018; Kim et al., 2018; Marino et al., 2018). The most related example is SA-VAE (Kim et al., 2018), which mixes instance-specific variational inference and empirically avoids posterior collapse. Our approach is much simpler without sacrificing performance, yet achieves an average of  $5\mathrm{x}$  training speedup.

Other attempts to address posterior collapse include proposing new regularizers (Zhao et al., 2017; Tolstikhin et al., 2018; Phuong et al., 2018), deploying less powerful decoders (Yang et al., 2017; Semeniuta et al., 2017), using lossy input (Chen et al., 2017), utilizing skip connections (Dieng et al., 2017; 2018), or changing the prior (Tomczak & Welling, 2018; Xu & Durrett, 2018).

# 6 EXPERIMENTS

Our experiments below are designed to (1) examine whether the proposed method indeed prevents posterior collapse, (2) test its efficacy with respect to maximizing predictive log-likelihood compared to other existing approaches, and (3) test its training efficiency.

# 6.1 SETUP

For all experiments we use a Gaussian prior  $\mathcal{N}(\mathbf{0},\mathbf{I})$  and the inference network parametrizes a diagonal Gaussian. We evaluate with approximate negative log likelihood (NLL) as estimated by 500

Table 1: Results on all datasets. For LSTM-LM* and PixelCNN* we report the exact negative log likelihood.  

<table><tr><td rowspan="2">Model</td><td colspan="3">Yahoo</td><td colspan="3">Yelp15</td><td colspan="3">OMNIGLOT</td></tr><tr><td>NLL</td><td>KL</td><td>MI</td><td>NLL</td><td>KL</td><td>MI</td><td>NLL</td><td>KL</td><td>MI</td></tr><tr><td colspan="10">Previous Reports</td></tr><tr><td>VLAE (Chen et al., 2017)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>89.83</td><td>-</td><td>-</td></tr><tr><td>VampPrior (Tomczak &amp; Welling, 2018)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>89.76</td><td>-</td><td>-</td></tr><tr><td>CNN-VAE (Yang et al., 2017)</td><td>≤332.1</td><td>10.0</td><td>-</td><td>≤359.1</td><td>7.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SA-VAE + anneal (Kim et al., 2018)</td><td>≤327.5</td><td>7.19</td><td>-</td><td>-</td><td>-</td><td>-</td><td>≤90.05</td><td>2.78</td><td>-</td></tr><tr><td colspan="10">Modified VAE Objective</td></tr><tr><td>VAE + anneal</td><td>328.6</td><td>0.0</td><td>0.0</td><td>358.1</td><td>0.0</td><td>0.0</td><td>89.20</td><td>2.1</td><td>1.9</td></tr><tr><td>β-VAE (β = 0.2)</td><td>333.1</td><td>18.7</td><td>3.4</td><td>360.4</td><td>11.3</td><td>3.3</td><td>105.19</td><td>69.1</td><td>3.9</td></tr><tr><td>β-VAE (β = 0.4)</td><td>328.5</td><td>3.8</td><td>2.2</td><td>358.8</td><td>4.0</td><td>1.8</td><td>95.89</td><td>28.0</td><td>3.9</td></tr><tr><td>β-VAE (β = 0.6)</td><td>328.6</td><td>0.1</td><td>0.1</td><td>358.1</td><td>0.1</td><td>0.1</td><td>92.16</td><td>20.4</td><td>3.9</td></tr><tr><td>β-VAE (β = 0.8)</td><td>328.8</td><td>0.0</td><td>0.0</td><td>357.9</td><td>0.0</td><td>0.0</td><td>89.27</td><td>9.6</td><td>3.9</td></tr><tr><td>SA-VAE + anneal</td><td>327.4</td><td>3.5</td><td>1.9</td><td>355.9</td><td>2.3</td><td>1.3</td><td>89.01</td><td>3.4</td><td>2.6</td></tr><tr><td>Ours + anneal</td><td>326.6</td><td>6.7</td><td>3.2</td><td>355.9</td><td>3.7</td><td>2.3</td><td>89.12</td><td>2.5</td><td>2.2</td></tr><tr><td colspan="10">Standard VAE Objective</td></tr><tr><td>LSTM-LM*</td><td>328.1</td><td>-</td><td>-</td><td>357.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PixelCNN*</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>89.76</td><td>-</td><td>-</td></tr><tr><td>VAE</td><td>328.8</td><td>0.0</td><td>0.0</td><td>358.2</td><td>0.0</td><td>0.0</td><td>89.41</td><td>1.5</td><td>1.4</td></tr><tr><td>SA-VAE</td><td>329.1</td><td>0.1</td><td>0.0</td><td>357.7</td><td>0.4</td><td>0.3</td><td>89.30</td><td>2.6</td><td>2.2</td></tr><tr><td>Ours</td><td>328.0</td><td>5.4</td><td>3.0</td><td>357.0</td><td>3.8</td><td>2.6</td><td>89.03</td><td>2.5</td><td>2.2</td></tr></table>

importance weighted samples (Burda et al., 2016) since it produces a tighter lower bound to marginal data log likelihood than ELBO, and should be more accurate. We also report  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x})\| p(\mathbf{z}))$  (KL), mutual information  $I_{q}$  (MI) and include ELBO in Appendix C.

As baselines, we compare with strong neural autoregressive models (LSTM-LM for text and PixelCNN (van den Oord et al., 2016) for images), basic VAE, the "KL cost annealing" method (Bowman et al., 2016; Sønderby et al., 2016),  $\beta$ -VAE (Higgins et al., 2017), and SA-VAE (Kim et al., 2018) which holds the previous state-of-the-art performance on text modeling benchmarks. For  $\beta$ -VAE we vary  $\beta$  between 0.2, 0.4, 0.6, and 0.8. SA-VAE is ran with 10 refinement steps. We also examine the effect of KL cost annealing on both SA-VAE and our approach. To facilitate our analysis later, we report the results in two categories: "Standard VAE objectives", and "Modified VAE objectives".<sup>4</sup>

We evaluate our method on density estimation for text on the Yahoo and Yelp corpora (Yang et al., 2017) and images on OMNIGLOT (Lake et al., 2015). Following the same configuration as in Kim et al. (2018), we use a single layer LSTM as encoder and decoder for text. For images, we use a ResNet (He et al., 2016) encoder and a 13-layer Gated PixelCNN (van den Oord et al., 2016) decoder. We use 32-dimensional  $\mathbf{z}$  and optimize ELBO objective with SGD for text and Adam (Kingma & Ba, 2015) for images. We concatenate  $\mathbf{z}$  to the input for the decoders. For text,  $\mathbf{z}$  also predicts the initial hidden state of the LSTM decoder. We dynamically binarize images during training and test on fixed binarized test data. Full details for setup are in Appendix B.2 and B.3.

# 6.2 RESULTS

In Table 1 we show the results on all three datasets. Our method outperforms all baselines for the three datasets in the "standard" as well as the "modified" section on text. We observe that SA-VAE suffers from posterior collapse on both text datasets without annealing. However, we demonstrate that our algorithm does not experience posterior collapse without annealing.

Note that to examine posterior collapse issue for images we use a larger PixelCNN decoder than previous work, thus our approach is not directly comparable to them and included at the top of Table 1 as reference points. On OMNIGLOT our method without annealing achieves comparable performance to SA-VAE combined with annealing.

![](images/dec26dd9be0fbb70f6f6d130eb35b3f5f74c659d849ab84071a4ca058d6b2802.jpg)  
Figure 4: Training behavior on Yelp. Left: VAE + annealing. Middle: Our method. Right:  $\beta$ -VAE ( $\beta = 0.2$ ).

![](images/9b4ce4399f61cfaf8d7c52a982e5a669f3f9fc4bdc0980775f569d6b652201b7.jpg)

![](images/246738e1d91fcc7329fa73174a8921bc88d0ad393defb151a79b0963030dbe9e.jpg)

Table 2: Comparison of total training time, in terms of relative speed and absolute hours.  

<table><tr><td></td><td colspan="2">Yahoo</td><td colspan="2">Yelp15</td><td colspan="2">Omniglot</td></tr><tr><td></td><td>Relative</td><td>Hours</td><td>Relative</td><td>Hours</td><td>Relative</td><td>Hours</td></tr><tr><td>VAE</td><td>1.00</td><td>5.35</td><td>1.00</td><td>5.75</td><td>1.00</td><td>4.30</td></tr><tr><td>SA-VAE</td><td>9.91</td><td>52.99</td><td>10.33</td><td>59.37</td><td>15.15</td><td>65.07</td></tr><tr><td>Ours</td><td>2.20</td><td>11.76</td><td>3.73</td><td>21.44</td><td>2.19</td><td>9.42</td></tr></table>

# 6.3 ANALYSIS

We analyze the difference between our approach and the methods that weaken the KL regularizer term in ELBO, and explain the unwanted behavior produced by breaking maximum likelihood estimation. As illustrative examples, we compare with the KL cost annealing method and  $\beta$ -VAE. Decreasing the weight of the KL regularizer term in ELBO is equivalent to adding an additional regularizer to push  $q_{\phi}(\mathbf{z}|\mathbf{x})$  far from  $p(\mathbf{z})$ . We set  $\beta = 0.2$  in order to better observe this phenomenon.

We investigate the training procedure on the Yelp dataset based on: (1) the mutual information between  $\mathbf{z}$  and  $\mathbf{x}$ ,  $I_{q}$ , (2) the KL regularizer,  $\mathbb{E}_{\mathbf{x} \sim p_{d}(\mathbf{x})}[D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}))]$ , and (3) the distance between the aggregated posterior and the prior,  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z})\|p(\mathbf{z}))$ . Note that the KL regularizer is equal to the sum of the other two as stated in Eq.5. We plot these values over the course of training in Figure 4. In the initial training stage we observe that the KL regularizer increases with all three approaches, however, the mutual information,  $I_{q}$ , in the annealing remains small, thus a large KL regularizer term does not imply that the latent variable is being used. Finally the annealing method suffers from posterior collapse. For  $\beta$ -VAE, the mutual information increases, but  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z})\|p(\mathbf{z}))$  also reaches a very large value. Intuitively,  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z})\|p(\mathbf{z}))$  should be kept small for learning the generative model well since in the objective the generator  $p_{\theta}(\mathbf{x}|\mathbf{z})$  is learned with latent variables sampled from the variational distribution. If the setting of  $\mathbf{z}$  that best explains the data has a lower likelihood under the model prior, then the overall model would fit the data poorly. The same intuition has been discussed in Zhao et al. (2017) and Tolstikhin et al. (2018). This also explains why  $\beta$ -VAE generalizes poorly when it has large mutual information. In contrast, our approach is able to obtain high mutual information, and at the same time maintain a small  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z})\|p(\mathbf{z}))$  as a result of optimizing standard ELBO where the KL regularizer upper-bounds  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z})\|p(\mathbf{z}))$ .

# 6.4 TIMING

In Table 2 we report the total training time of our approach, SA-VAE and basic VAE-training across the three datasets. We find that the training time for our algorithm is only 2-3 times slower than a regular VAE whilst being 3-7 times faster than SA-VAE.

# 7 CONCLUSION

In this paper we study the "posterior collapse" problem that variational autoencoders experience when the model is parameterized by a strong autoregressive neural network. In our synthetic experiment we identify that the problem lies with the lagging inference network in the initial stages of training. To remedy this, we propose a simple yet effective training algorithm that aggressively optimizes the inference network with more updates before reverting back to basic VAE-training. Experiments on text and image modeling demonstrate effectiveness of our approach.

# REFERENCES

Alexander Alemi, Ben Poole, Ian Fischer, Joshua Dillon, Rif A Saurous, and Kevin Murphy. Fixing a broken elbo. In Proceedings of ICML, 2018.  
Samuel R Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. In Proceedings of SIGNLL, 2016.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. In Proceedings of ICLR, 2016.  
Xi Chen, Diederik P Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, and Pieter Abbeel. Variational lossy autoencoder. In Proceedings of ICLR, 2017.  
Chris Cremer, Xuechen Li, and David Duvenaud. Inference suboptimality in variational autoencoders. In Proceedings of ICML, 2018.  
Adji B Dieng, Chong Wang, Jianfeng Gao, and John Paisley. Topicrn: A recurrent neural network with long-range semantic dependency. In Proceedings of ICLR, 2017.  
Adji B Dieng, Yoon Kim, Alexander M Rush, and David M Blei. Avoiding latent variable collapse with generative skip models. In Proceedings of ICML workshop on Theoretical Foundations and Applications of Deep Generative Models, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of CVPR, 2016.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In Proceedings of ICLR, 2017.  
Devon Hjelm, Ruslan R Salakhutdinov, Kyunghyun Cho, Nebojsa Jojic, Vince Calhoun, and Junyoung Chung. Iterative refinement of the approximate posterior for directed belief networks. In Proceedings of NIPS, 2016.  
Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic variational inference. The Journal of Machine Learning Research, 14(1):1303-1347, 2013.  
Yoon Kim, Sam Wiseman, Andrew C Miller, David Sontag, and Alexander M Rush. Semi-amortized variational autoencoders. In Proceedings of ICML, 2018.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of ICLR, 2015.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings of ICLR, 2014.  
Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Proceedings of NIPS, 2016.  
Rahul Krishnan, Dawen Liang, and Matthew Hoffman. On the challenges of learning with inference networks on sparse, high-dimensional data. In Proceedings of AISTATS, 2018.  
Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332-1338, 2015.  
Joseph Marino, Yisong Yue, and Stephan Mandt. Iterative amortized inference. In Proceedings of ICML, 2018.  
Mary Phuong, Max Welling, Nate Kushman, Ryota Tomioka, and Sebastian Nowozin. The mutual autoencoder: Controlling information in latent code representations, 2018. URL https://openreview.net/forum?id=HkbmWqxCZ.  
Stanislau Semeniuta, Aliaksei Severyn, and Erhardt Barth. A hybrid convolutional variational autoencoder for text generation. In Proceedings of EMNLP, 2017.

Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. Ladder variational autoencoders. In Proceedings of NIPS, 2016.  
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein autoencoders. In Proceedings of ICLR, 2018.  
Jakub M. Tomczak and Max Welling. Vae with a vampprior. In Proceedings of AISTATS, 2018.  
Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelCNN decoders. In Proceedings of NIPS, 2016.  
Jiacheng Xu and Greg Durrett. Spherical latent spaces for stable variational autoencoders. In Proceedings of EMNLP, 2018.  
Zichao Yang, Zhiting Hu, Ruslan Salakhutdinov, and Taylor Berg-Kirkpatrick. Improved variational autoencoders for text modeling using dilated convolutions. In Proceedings of ICML, 2017.  
Shengjia Zhao, Jiaming Song, and Stefano Ermon. Infovae: Information maximizing variational autoencoders. arXiv preprint arXiv:1706.02262, 2017.
