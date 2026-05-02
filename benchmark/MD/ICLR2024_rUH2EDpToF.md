# GENERATIVE MARGINALIZATION MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce marginalization models (MAMs), a new family of generative models for high-dimensional discrete data. They offer scalable and flexible generative modeling with tractable likelihoods by explicitly modeling all induced marginal distributions. Marginalization models enable fast evaluation of arbitrary marginal probabilities with a single forward pass of the neural network, which overcomes a major limitation of methods with exact marginal inference, such as autoregressive models (ARMs). We propose scalable methods for learning the marginals, grounded in the concept of "marginalization self-consistency". Unlike previous methods, MAMs also support scalable training of any-order generative models for high-dimensional problems under the setting of energy-based training, where the goal is to match the learned distribution to a given desired probability (specified by an unnormalized (log) probability function such as energy or reward function). We demonstrate the effectiveness of the proposed model on a variety of discrete data distributions, including binary images, language, physical systems, and molecules, for maximum likelihood and energy-based training settings. MAMs achieve orders of magnitude speedup in evaluating the marginal probabilities on both settings. For energy-based training tasks, MAMs enable any-order generative modeling of high-dimensional problems beyond the capability of previous methods.

# 1 INTRODUCTION

Deep generative models have enabled remarkable progress across diverse fields, including image generation, audio synthesis, natural language modeling, and scientific discovery. However, there remains a pressing need to better support efficient probabilistic inference for key questions involving marginal probabilities  $p(\mathbf{x}_s)$  and conditional probabilities  $p(\mathbf{x}_u|\mathbf{x}_v)$ , for appropriate subsets  $s, u, v$  of the variables. The ability to directly address such quantities is critical in applications such as outlier detection [50, 40], masked language modeling [11, 72], image inpainting [73], and constrained protein/molecule design [69, 55]. Furthermore, the capacity to conduct such inferences for arbitrary subsets of variables empowers users to leverage the model according to their specific needs and preferences. For instance, in protein design, scientists may want to manually guide the generation of a protein from a user-defined substructure under a particular path over the relevant variables. This requires the generative model to perform arbitrary marginal inferences.

Towards this end, neural autoregressive models (ARMs) [3, 30] have been developed to facilitate conditional/marginal inference based on the idea of modeling a high-dimensional joint distribution as a factorization of univariate conditionals using the chain rule of probability. Many efforts have been made to scale up ARMs and enable any-order generative modeling under the setting of maximum likelihood estimation (MLE) [30, 66, 20], and great progress has been made in applications such as masked language modeling [72] and image inpainting [20]. However, marginal likelihood evaluation in the most widely-used modern neural network architectures (e.g., Transformers [68] and U-Nets [53]) is limited by  $\mathcal{O}(D)$  neural network passes, where  $D$  is the length of the sequence. This scaling makes it difficult to evaluate likelihoods on long sequences arising in data such as natural language and proteins. In contrast to MLE, in the setting of energy-based training (EB), instead of empirical data samples, we only have access to an unnormalized (log) probability function (specified by a reward or energy function) that can be evaluated pointwise for the generative model to match. In such settings, ARMs are limited to fixed-order generative modeling and lack scalability in training. The subsampling techniques developed to scale the training of conditionals for MLE are no longer applicable when matching log probabilities in energy-based training (see Section 4.3 for details).

![](images/9aa060974f5a2005fcfa43707e57dca6fe008e5cb2ffe044b981b1981805a14c.jpg)  
Figure 1: Marginalization models (MAMs) enable estimation of any marginal probability with a neural network  $\theta$  that learns to "marginalize out" variables. The figure illustrates marginalization of a single variable on bit strings (representing molecules) with two alternatives (versus  $K$  in general) for clarity. The bars represent probability masses.

To enhance scalability and flexibility in the generative modeling of discrete data, we propose a new family of generative models, **marginalization models** (MAMs), that directly model the marginal distribution  $p(\mathbf{x}_s)$  for any subset of variables  $\mathbf{x}_s$  in  $\mathbf{x}$ . Direct access to marginals has two important advantages: 1) significantly speeding up inference for any marginal, and 2) enabling scalable training of any-order generative models under both MLE and EB settings.

The unique structure of the model allows it to simultaneously represent the coupled collection of all marginal distributions of a given discrete joint probability mass function. For the model to be valid, it must be consistent with the sum rule of probability, a condition we refer to as "marginalization self-consistency" (see Figure 1); learning to enforce this with scalable training objectives is one of the key contributions of this work.

We show that MAMs can be trained under both maximum likelihood and energy-based training settings with scalable learning objectives. We demonstrate the effectiveness of MAMs in both settings on a variety of discrete data distributions, including binary images, text, physical systems, and molecules. We empirically show that MAMs achieve orders of magnitude speedup

![](images/275ba2cb8b2b586dd9c763bdbb1eb0f4fc5abb3268d383b7ffc580ccc6517b05.jpg)  
Figure 2: Scalability of sequential discrete generative models. The y-axis unit is # of NN forward passes required.

in marginal likelihood evaluation. For energy-based training, MAMs are able to scale training of any-order generative models to high-dimensional problems that previous methods fail to achieve.

# 2 BACKGROUND

We first review two prevalent generative modeling settings. Then we introduce autoregressive models under two training settings.

Maximum likelihood (MLE) Given a dataset  $\mathcal{D} = \{\mathbf{x}^{(i)}\}_{i=1}^{N}$  drawn from a data distribution  $p = p_{\mathrm{data}}$ , we aim to learn the distribution  $p_{\theta}(\mathbf{x})$  that maximizes the probability of the data under our model. Mathematically, we aim to learn the parameters  $\theta^{\star}$  that maximize the log-likelihood:

$$
\theta^ {\star} = \arg \max  _ {\theta} \mathbb {E} _ {\mathbf {x} \sim p _ {\mathrm {d a t a}}} \left[ \log p _ {\theta} (\mathbf {x}) \right] \approx \arg \max  _ {\theta} 1 / N \sum_ {i = 1} ^ {N} \log p _ {\theta} (\mathbf {x} ^ {(i)}) \tag {1}
$$

which is also equivalent to minimizing the Kullback-Leibler divergence under the empirical distribution, i.e., minimizing  $D_{\mathrm{KL}}(p_{\mathrm{data}}(\mathbf{x})||p_{\theta}(\mathbf{x}))$ . This is the setting that is most commonly used in generation of images (e.g., diffusion models [59, 18, 60]) and language (e.g. GPT [49]) where we can empirically draw observed data from the distribution.

Energy-based training (EB) In this setting, we do not have data from the distribution of interest. Instead, we have access to the unnormalized  $(\log)$  probability mass function  $f$ , usually in the form of reward function or energy function, that are defined by humans or by physical systems to specify how likely a sample is. Mathematically, we can define the target probability mass function to be  $f(\mathbf{x}) = \exp(r(x) / \tau)$ , where  $r(x)$  is the reward function and  $\tau > 0$  is a temperature parameter. This expresses the intuitive idea that we would like the model to assign higher probability to data with larger reward. For example, the reward function can represent human preferences in alignment of large language models [43, 42]. In molecular/material design applications, scientists can specify the reward according to how close a particular sample's measured or calculated properties are to

some functional desiderata. When modeling the thermodynamic ensemble of physical systems,  $r(x)$  is defined to be the (negative) energy function of a given state [41]. Mathematically, we aim to learn the parameters  $\theta$  such that  $p_{\theta}(\mathbf{x}) \approx f(\mathbf{x}) / Z$ , where  $Z$  is the normalization constant of  $f$ . A common training criteria is to minimize the KL divergence [41, 71, 9]:

$$
\min  _ {\theta} D _ {\mathrm {K L}} \left(p _ {\theta} (\mathbf {x}) \| f (\mathbf {x}) / z\right) = \mathbb {E} _ {\mathbf {x} \sim p _ {\theta} (\mathbf {x})} \left[ \log p _ {\theta} (\mathbf {x}) - \log f (\mathbf {x}) / z \right]. \tag {2}
$$

Autoregressive models Autoregressive models (ARMs) [3, 30] model a complex high-dimensional distribution  $p(\mathbf{x})$  by factorizing it into univariate conditionals using the chain rule:

$$
\log p (\mathbf {x}) = \sum_ {d = 1} ^ {D} \log p \left(x _ {d} \mid \mathbf {x} _ {<   d}\right), \tag {3}
$$

where  $\mathbf{x}_{<d} = \{x_1, \ldots, x_{d-1}\}$ . Recently there has been great success in applying autoregressive models to discrete data, such as natural language, proteins [58, 32, 36], and molecules [56, 15]. Due to their sequential nature via modeling the conditionals, evaluation of (joint/marginal) likelihood requires up to  $D$  neural network evaluations. This is costly for long sequences, leading to limitations that prevent ARMs to be scalable for marginal inference and energy-based training.

Any-order ARMs (AO-ARMs) Under the MLE setting, Uria et al. [66] propose to learn the conditionals of ARMs for arbitrary orderings that include all permutations of  $\{1,\dots ,D\}$ . The model  $\phi$  can be trained by maximizing a lower-bound objective [66, 20] that takes an expectation under a uniform distribution on orderings. This objective allows scalable training of AO-ARMs, leveraging efficient parallel evaluation of multiple one-step conditionals for each token in one forward pass with architectures such as the U-Net [53] and Transformers [68]. However, under the EB setting, training AO-ARMs presents challenges, which we will discuss in details in Section 4.3.

# 3 MARGINALIZATION MODELS

We propose marginalization models (MAMs), a new type of generative model that enables scalable any-order generative modeling as well as efficient marginal evaluation, for both maximum likelihood and energy-based training. The flexibility and scalability of marginalization models are enabled by the explicit modeling of the marginal distribution and enforcing marginalization self-consistency.

In this paper, we focus on generative modeling of discrete structures using vectors of discrete variables. The vector representation encompasses various real-world problems with discrete structures, including language sequence modeling, protein design, and molecules with string-based representations (e.g., SMILES [70] and SELFIES [29]). Moreover, vector representations are inherently applicable to any discrete problem, since it is feasible to encode any discrete object into a vector of discrete variables.

Definition We are interested in modeling the discrete probability distribution  $p(\mathbf{x})$ , where  $\mathbf{x} = [x_1, \ldots, x_D]$  is a  $D$ -dimensional vector and each  $x_d$  takes  $K$  possible values, i.e.  $x_d \in \{1, \ldots, K\}$ .

Marginalization Let  $\mathbf{x}_s$  be a subset of variables of  $\mathbf{x}$  and  $\mathbf{x}_{s^c}$  be the complement set, i.e.  $\mathbf{x}_s\subseteq$ $\{x_{1},\ldots ,x_{D}\}$  and  $\mathbf{x}_{s^c} = \{x_1,\dots ,x_D\} \setminus \mathbf{x}_s$ . The marginal of  $\mathbf{x}_s$  is obtained by summing over all values of  $\mathbf{x}_{s^c}$ :

$$
p \left(\mathbf {x} _ {s}\right) = \sum_ {\mathbf {x} _ {s ^ {c}}} p \left(\mathbf {x} _ {s}, \mathbf {x} _ {s ^ {c}}\right) \tag {4}
$$

We refer to (4) as the "marginalization self-consistency" that any valid distribution should follow. The goal of a marginalization model  $\theta$  is to estimate the marginals  $p(\mathbf{x}_s)$  for any subset of variables  $\mathbf{x}_s$  as closely as possible. To achieve this, we train a deep neural network  $p_{\theta}$  that minimizes the distance of  $p_{\theta}(\mathbf{x})$  and  $p(\mathbf{x})$  on the full joint distribution<sup>1</sup> while enforcing the marginalization self-consistency.

Parameterization To approximate arbitrary marginals over  $\mathbf{x}_s$  with a single neural network forward pass, we additionally include the "marginalized out" variables  $\mathbf{x}_{s^c}$  in the input by introducing a special symbol  $n$  to denote the missing values. By doing this, we create an augmented  $D$ -dimensional vector representation  $\mathbf{x}_s^{\mathrm{aug}} \in \mathcal{X}^{\mathrm{aug}} \triangleq \{1, \dots, K, ?\}^D$  and feed it to the NN. For example, for a binary vector  $\mathbf{x}$  of length 4, for  $\mathbf{x}_s = \{x_1, x_3\}$  with  $x_1 = 0$  and  $x_3 = 1$ ,  $\mathbf{x}_s^{\mathrm{aug}} = [0, ?, 1, ?]$  where  $n$  denotes  $x_2$  and  $x_4$  being marginalized out. From here onwards we will use  $\mathbf{x}_s^{\mathrm{aug}}$  and  $\mathbf{x}_s$  interchangeably.

A marginalization model parameterized by a neural network  $\theta$  takes in the augmented vector representation  $\mathbf{x}_s^{\mathrm{aug}}\in \{1,\dots ,K,?\} ^D$ , and outputs the marginal log probability  $f_{\theta}(\mathbf{x}_s) = \log p_{\theta}(\mathbf{x}_s)$  that satisfy the marginalization self-consistency constraints:

$$
\sum_ {\mathbf {x} _ {s ^ {c}}} p _ {\theta} ([ \mathbf {x} _ {s}, \mathbf {x} _ {s ^ {c}} ]) = p _ {\theta} (\mathbf {x} _ {s}) \quad \forall \mathbf {x} _ {s} \in \{1, \ldots , K, ? \} ^ {D}
$$

where  $[\mathbf{x}_s,\mathbf{x}_{s^c}]$  denotes the concatenation of  $\mathbf{x}_s$  and  $\mathbf{x}_{s^c}$ . Given a random ordering of the variables  $\sigma \in S_D$  where  $S_{D}$  defines the set of all permutations of  $1,2,\dots ,D$ , let  $\sigma (d)$  denote the  $d$ -th element in  $\sigma$  and  $\sigma (< d)$  be the first  $d - 1$  elements in  $\sigma$ . The marginalization can be imposed over one variable at a time, which leads to the following one-step marginalization constraints:

$$
p _ {\theta} \left(\mathbf {x} _ {\sigma (<   d)}\right) = \sum_ {x _ {\sigma (d)}} p _ {\theta} \left(\mathbf {x} _ {\sigma (\leq d)}\right), \quad \forall \sigma \in S _ {D}, \mathbf {x} \in \{1, \dots , K \} ^ {D}, d \in [ 1: D ]. \tag {5}
$$

Sampling Given the learned marginalization model, one can sample from the learned distribution by picking an arbitrary order  $\sigma$  and sampling one variable at a time. To evaluate the conditionals at each step of the generation, we can use the product rule of probability:

$$
p _ {\theta} \left(x _ {\sigma (d)} \mid \mathbf {x} _ {\sigma (<   d)}\right) = p _ {\theta} \left(\mathbf {x} _ {\sigma (\leq d)}\right) / p _ {\theta} \left(\mathbf {x} _ {\sigma (<   d)}\right).
$$

However, the above is not a valid conditional distribution if the marginalization in (5) is not strictly enforced, since it might not sum up exactly to one. Hence we use following normalized conditional:

$$
p _ {\theta} \left(x _ {\sigma (d)} \mid \mathbf {x} _ {\sigma (<   d)}\right) = \frac {p _ {\theta} \left(\left[ \mathbf {x} _ {\sigma (<   d)} , x _ {\sigma (d)} \right]\right)}{\sum_ {x _ {\sigma (d)}} p _ {\theta} \left(\left[ \mathbf {x} _ {\sigma (<   d)}, x _ {\sigma (d)} \right]\right)}. \tag {6}
$$

In this paper, we focus on the sampling procedure that generates one variable at a time, but marginalization models can also facilitate sampling multiple variables at a time (See Appendix B.2).

Scalable learning of marginals with conditionals In training, we impose the marginalization self-consistency by minimizing the squared error of the constraints in (5) in log-space. Evaluation of each marginalization constraint in (5) requires  $K$  NN forward passes, where  $K$  is the number of discrete values  $x_{d}$  can take. This makes training challenging to scale when  $K$  is large. To address this issue, we augment the marginalization models with learnable conditionals parameterized by  $\phi$ . The marginalization constraints in (5) can be decomposed into  $K$  parallel marginalization constraints, which makes it highly scalable to subsample from for training:

$$
p _ {\theta} \left(\mathbf {x} _ {\sigma (<   d)}\right) p _ {\phi} \left(\mathbf {x} _ {\sigma (d)} \mid \mathbf {x} _ {\sigma (<   d)}\right) = p _ {\theta} \left(\mathbf {x} _ {\sigma (\leq d)}\right), \quad \forall \sigma \in S _ {D}, \mathbf {x} \in \{1, \dots , K \} ^ {D}, d \in [ 1: D ]. \tag {7}
$$

During training, we need to specify a distribution  $q(\mathbf{x})$  for subsampling the marginalization constraints to optimize on. In practice, it can be set to the distribution we are interested to perform marginal inference on, such as  $p_{\mathrm{data}}$  or the distribution of the generative model  $p_{\theta, \phi}$ .

# 4 TRAINING THE MARGINALIZATION MODELS

# 4.1 MAXIMUM LIKELIHOOD ESTIMATION TRAINING

In this setting, we train MAMs with the maximum likelihood objective while additionally enforcing the marginalization constraints in Equation (5):

$$
\max  _ {\theta , \phi} \mathbb {E} _ {\mathbf {x} \sim p _ {\mathrm {d a t a}}} \log p _ {\theta} (\mathbf {x}) \tag {8}
$$

$$
\begin{array}{l} \text {s . t .} p _ {\theta} (\mathbf {x} _ {\sigma (<   d)}) p _ {\phi} (\mathbf {x} _ {\sigma (d)} | \mathbf {x} _ {\sigma (<   d)}) = p _ {\theta} (\mathbf {x} _ {\sigma (\leq d)}), \forall \sigma \in S _ {D}, \mathbf {x} \in \{1, \dots , K \} ^ {D}, d \in [ 1: D ]. \end{array}
$$

Two-stage training A typical way to solve the above optimization problem is to convert the constraints into a penalty term and optimize the penalized objective, but we empirically found the learning to be slow and unstable. Instead, we identify an alternative two-stage optimization formulation that is theoretically equivalent to Equation (8), but leads to more efficient training:

Claim 1. Solving the optimization problem in (8) is equivalent to the following two-stage optimization procedure, under mild assumption about the neural networks used being universal approximators:

Stage 1:  $\max_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{data}} \mathbb{E}_{\sigma \sim \mathcal{U}(S_D)} \sum_{d=1}^{D} \log p_{\phi}\left(x_{\sigma(d)} \mid \mathbf{x}_{\sigma(<d)}\right)$

Stage 2:  $\min_{\theta} \mathbb{E}_{\mathbf{x} \sim q(\mathbf{x})} \mathbb{E}_{\sigma \sim \mathcal{U}(S_D)} \mathbb{E}_{d \sim \mathcal{U}(1, \dots, D)} \left( \log[p_{\theta}(\mathbf{x}_{\sigma(<d)}) p_{\phi}(\mathbf{x}_{\sigma(d)} | \mathbf{x}_{\sigma(<d)})] - \log p_{\theta}(\mathbf{x}_{\sigma(\leq d)}) \right)^2.$

To make sure  $p_{\theta}$  is normalized, we can either additionally enforce  $p_{\theta}([??\dots ?]) = 1$  or let  $Z_{\theta} = p_{\theta}([??\dots ?])$  be the normalization constant.

The first stage can be interpreted as fitting the conditionals in the same way as AO-ARMs [66, 20] and the second stage acts as distilling the marginals from conditionals. The intuition comes from the chain rule of probability: there is a one-to-one correspondence between optimal conditionals  $\phi$  and marginals  $\theta$ , i.e.  $\log p_{\theta}(\mathbf{x}) = \sum_{d=1}^{D} \log p_{\phi}(x_{\sigma(d)} | \mathbf{x}_{\sigma(<d)})$  for any  $\sigma$  and  $\mathbf{x}$ . By assuming neural networks are universal approximators, we can first optimize for the optimal conditionals, and then optimize for the corresponding optimal marginals. We provide more details in Appendix A.1.

# 4.2 ENERGY-BASED TRAINING

In this setting, we train MAMs using the energy-based training objective in Equation (2) with a penalty term to enforce the marginalization constraints in Equation (5):

$$
\min _ {\theta , \phi} D _ {\mathrm {K L}} (p _ {\theta} (\mathbf {x}) \| p (\mathbf {x})) + \lambda \mathbb {E} _ {\mathbf {x} \sim q (\mathbf {x})} \mathbb {E} _ {\sigma} \mathbb {E} _ {d} \bigl (\log \bigl [ p _ {\theta} (\mathbf {x} _ {\sigma (<   d)}) p _ {\phi} (\mathbf {x} _ {\sigma (d)} | \mathbf {x} _ {\sigma (<   d)}) \bigr ] - \log p _ {\theta} (\mathbf {x} _ {\sigma (\leq d)}) \bigr) ^ {2},
$$

where  $\sigma \sim \mathcal{U}(S_D)$ ,  $d \sim \mathcal{U}(1, \dots, D)$  and  $q(\mathbf{x})$  is the distribution of interest for evaluating marginals.

Scalable training We use REINFORCE to estimate the gradient of the KL divergence term:

$$
\begin{array}{l} \nabla_ {\theta} D _ {\mathrm {K L}} (p _ {\theta} (\mathbf {x}) | | p (\mathbf {x})) = \mathbb {E} _ {\mathbf {x} \sim p _ {\theta} (\mathbf {x})} \left[ \nabla_ {\theta} \log p _ {\theta} (\mathbf {x}) \left(\log p _ {\theta} (\mathbf {x}) - \log f (\mathbf {x})\right) \right] \\ \approx 1 / N \sum_ {i = 1} ^ {N} \nabla_ {\theta} \log p _ {\theta} (\mathbf {x} ^ {(i)}) (\log p _ {\theta} (\mathbf {x} ^ {(i)}) - \log f (\mathbf {x} ^ {(i)})) \tag {9} \\ \end{array}
$$

For the penalty term, we subsample the ordering  $\sigma$  and step  $d$  for each data  $\mathbf{x}$ .

Efficient sampling with persistent MCMC We need cheap and effective samples from  $p_{\theta}$  in order to perform REINFORCE, so a persistent set of Markov chains are maintained by randomly picking an ordering and taking block Gibbs sampling steps using the conditional distribution  $p_{\phi}(\mathbf{x}_{\sigma (d)}|\mathbf{x}_{\sigma (< d)})$  (full algorithm in Appendix A.5), in similar fashion to persistent contrastive divergence [64]. The samples from the conditional distribution  $p_{\phi}$  serve as approximate samples from  $p_{\theta}$  when they are close to each other. Otherwise, we can additionally use importance sampling for adjustment.

# 4.3 ADDRESSING LIMITATIONS OF ARMS

We discuss in more detail about how MAMs address some limitations of ARMs. The first one is general to both training settings, while the latter two are specific to energy-based training.

1) Slow marginal inference of likelihoods Due to sequential conditional modeling, evaluation of a marginal  $p_{\phi}(\mathbf{x}_o)$  with ARMs (or an arbitrary marginal with AO-ARMs) requires applying the NN  $\phi$  up to  $D$  times, which is inefficient in time and memory for high-dimensional data. In comparison, MAMs are able to estimate any arbitrary marginal with one NN forward pass.  
2) Lack of support for any-order training In energy-based training, the objective in Equation (2) aims to minimize the distance between  $\log p_{\phi}(\mathbf{x})$  and  $\log p(x)$ , where  $\phi$  is the NN parameters of an ARM. However, unless the ARM is perfectly self-consistent over all orderings, it will not be the case that  $\log p_{\phi}(\mathbf{x}) = \mathbb{E}_{\sigma}\log p_{\phi}(\mathbf{x}|\sigma)$ .

Therefore, the expected  $D_{\mathrm{KL}}$  objective over the orderings

$\sigma$  would not be equivalent to the original  $D_{\mathrm{KL}}$  objective, i.e.,  $\mathbb{E}_{p_{\phi}}[\mathbb{E}_{\sigma}\log p_{\phi}(x|\sigma) - \log p(x)]\neq \mathbb{E}_{p_{\phi}}[\log p_{\phi}(x) - \log p(x)].$  As a result, ARMs cannot be trained with the expected  $D_{\mathrm{KL}}$  objective over all orderings simultaneously, but instead need to resort to a preset order and minimize the KL divergence between  $\log p_{\phi}(\mathbf{x}|\sigma)$  and the target density  $\log p(\mathbf{x})$  . The

self-consistency constraints imposed by MAMs address this issue. MAMs are not limited to fixed ordering because marginals are order-agnostic and we can optimize over expectation of orderings for the marginalization self-consistency constraints.

3) Training not scalable on high-dimensional problems When minimizing the difference between  $\log p_{\phi}(\mathbf{x}|\sigma)$  and the target  $\log p(\mathbf{x})$ , ARMs need to sum conditionals to evaluate  $\log p_{\phi}(\mathbf{x}|\sigma)$ . One might consider subsampling one-step conditionals  $p_{\phi}(x_{\sigma(d)}|\mathbf{x}_{\sigma(<d)})$  to estimate  $p_{\phi}(\mathbf{x})$ , but this leads

![](images/38b652e23887ea74e5069ae299ff5202d78e5b6e0f6c30f06e134ca4fcd4c4da.jpg)  
Figure 3: Approximating  $\log p_{\phi}(\mathbf{x})$  with one-step conditional (ARM-MC) results in extremely high gradient variance during energy-based training.

to high variance of the REINFORCE gradient in Equation (9) due to the product of the score function and distance terms, which are both high variance (We validate this in experiments, see Figure 3). Consequently, training ARMs for energy-based training necessitates a sequence of  $D$  conditional evaluations to compute the gradient of the objective function. This constraint leads to an effective batch size of  $B \times D$  for batch of  $B$  samples, significantly limiting the scalability of ARMs to high-dimensional problems. Furthermore, obtaining Monte Carlo samples from ARMs for the REINFORCE gradient estimator is slow when the dimension is high. Due to the fixed input ordering, this process requires  $D$  sequential sampling steps, making more cost-effective sampling approaches like persistent MCMC infeasible. Marginalization models circumvent this challenge by directly estimating the log-likelihood with the marginal neural network. Additionally, the support for any-order training enables efficient sampling through the utilization of persistent MCMC methods.

# 5 RELATED WORK

Autoregressive models Developments in deep learning have greatly advanced the performance of ARMs across different modalities, including images, audio, and text. Any-order (Order-agnostic) ARMs were first introduced in [66] by training with the any-order lower-bound objective for the maximum likelihood setting and recently seen in ARDM [20] with state-of-the-art performance for any-order discrete modeling of image/text/audio. Germain et al. [16] train an auto-encoder with masking that outputs the sequence of all one-step conditionals for a given ordering, but does not generate as well as methods [67, 72, 20] that predict one-step conditionals under the given masking. Douglas et al. [14] trains an AO-ARM and use importance sampling to estimate arbitrary conditional posteriors, but with limited experiment validation on a synthetic dataset. Shih et al. [57] utilizes a modified training objective of ARMs for better marginal inference performance but loses any-order generation capability. Comparisons of MAMs and ARMs are discussed in detail in Section 4.3.

Arbitrary conditional/marginal models For continuous data, VAEAC [25] and ACFlow [31] extends the idea of conditional variational encoder and normalizing flow to model arbitrary conditionals. ACE [62] improves the expressiveness of arbitrary conditional models through directly modeling the energy function, which puts less constraints on parameterization but comes at the cost of approximating the normalizing constant. Instead of using neural networks as function approximators, probabilistic circuits (PCs) [6, 45] offer tractable probabilistic models for both conditionals and marginals by building a computation graph with sum and product operations following specific structural constraints. Examples of PCs include Chow-Liu trees [7], arithmetic circuits [10], sum-product networks [47], etc. Peharz et al. [45] have improved the scalability of PCs through combining arithmetic operations into a single monolithic einsum-operation and automatic differentiation. More recently, [33, 34] demonstrated the potential of PCs with distilling latent variables from trained deep generative models on continuous image data. However, their expressiveness are limited by the structural constraints. All methods mentioned above focus on MLE settings, except ARMs are explored in energy-based training of science problems [9, 71], but suffer in scaling when  $D$  is large.

GFlowNets GFlowNets [2, 4] formulate the problem of generation as matching the probability flow at terminal states to the target normalized density. Compared to ARMs, GFlowNets allow flexible modeling of the generation process by assuming learnable generation paths through a directed acyclic graph (DAG). The advantages of learnable generation paths come with the trade-off of sacrificing the flexibility of any-order generation and exact likelihood evaluation. Under fixed generation path, GFlowNets are reduced to fixed-order ARMs [74]. In Appendix A.3, we further identify the connections and differences between GFlowNets and AO-ARMs/MAMs. For discrete problems, Zhang et al. [75] train GFlowNets on the squared distance loss with the trajectory balance objective [38], which is less scalable for large  $D$  (due to the same reason as ARMs in Section 4.3) and renders direct access to marginals unavailable. For the MLE setting, an energy function is additionally learned from data such that training is reduced to energy-based training.

# 6 EXPERIMENTS

We conduct experiments with marginalization models (MAM) on both MLE and EB settings for discrete problems including binary images, text, molecules and physical systems. We consider the following baselines for comparison: Any-order ARM (AO-ARM) [20], ARM [30], GFlowNet [39, 75], Discrete Flow[65] and Probabilistic Circuit (PC) [45]. MAM, PC and

![](images/97adac8a004831321419214ec788db89b5f581e99c7ca56974bcaff87cc05555.jpg)  
Original

![](images/b93cfea41882f3d03cf0452257077f43db41e983813fc1d78b7beda710429537.jpg)  
Censored-100

![](images/2c5effd224862dd8eca4a42e9618c1382a2be3bfc0e55a26ae74ff1b2a31aba7.jpg)  
Figure 4: An example of the data generated (with  $100 / 400 / 700$  pixels masked) for comparing the quality of likelihood estimate. Numbers below the images are LL estimates from MAM's marginal network (left) and AO-ARM-E's ensemble estimate (right).

![](images/6728d1442225ac6dde899d3528a7c60a9d1e4949363e120efdc94a0843fef993.jpg)  
Censored-400

![](images/dc4718518055ffb037a229048c8d00e588d77d4f97826dbd0a79bcb70075a8d4.jpg)  
Censored-700  
-54.48, -57.47

![](images/2d7dfe837188c574e5968b32cc92f352798782c28f966d08bd290367f69bbda3.jpg)  
Generated-100  
-60.48, -63.37

![](images/6bd87db0a21c24d08179fc66c81da42a1928eea916a74dc7024a1512e7052f82.jpg)  
Generated-400  
Generated-700  
-106.45, -108.58

Table 1: Performance Comparison on Binary-MNIST  

<table><tr><td>Model</td><td>NLL (bpd) ↓</td><td>Spearman&#x27;s ↑</td><td>Pearson ↑</td><td>Marg. inf. time (s) ↓</td></tr><tr><td>AO-ARM-E-U-Net</td><td>0.148</td><td>1.0</td><td>1.0</td><td>661.98 ± 0.49</td></tr><tr><td>AO-ARM-S-U-Net</td><td>0.149</td><td>0.996</td><td>0.993</td><td>132.40 ± 0.03</td></tr><tr><td>GflowNet-MLP</td><td>0.189</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PC-Image (EiNets)4</td><td>0.187</td><td>0.716</td><td>0.752</td><td>0.015 ± 0.00</td></tr><tr><td>MAM-U-Net</td><td>0.149</td><td>0.992</td><td>0.993</td><td>0.018 ± 0.00</td></tr></table>

(AO-)ARM support arbitrary marginal inference. Discrete flow<sup>3</sup> allows exact likelihood evaluation while GFlowNet needs to approximate the likelihood with sum using importance samples. For evaluating AO-ARM's marginal inference, we can either use an ensemble model by averaging over several random orderings (AO-ARM-E) or use a single random ordering (AO-ARM-S). In general, AO-ARM-E should always be better than AO-ARM-S but at a much higher cost. Neural network architecture and training hyperparameter details can be found in Appendix C.

Ablation studies on measuring marginal self-consistency and sampling with marginals are in Appendices B.1 and B.2.

Guidance on picking  $q$  is in Appendix B.3. Appendix C.3 contains more results on CIFAR-10.

# 6.1 MAXIMUM LIKELIHOOD ESTIMATION TRAINING

Binary MNIST We report the negative test likelihood (bits/digit), marginal estimate quality and marginal inference time per minibatch (of size 16) in Table (1). To keep GPU memory usage the same, we sequentially evaluate the likelihood for ARMs. Both MAM and AO-ARM use a U-Net architecture with 4 ResNet Blocks interleaved with attention layers (see Appendix C). GFlowNets fail to scale to large architectures as U-Net, hence we report GFlowNet results using an MLP from Zhang et al. [75]. For MAM, we use the conditional network to evaluate test likelihood (since this is also how MAM generates data). The marginal network is used for evaluating marginal inference. The quality of the marginal estimates will be compared to the best performing model.

In order to evaluate the quality of marginal likelihood estimates, we employ a controlled experiment where we randomly mask out portions of a test image and generate multiple samples with varying levels of masking (refer to Figure 4). This process allows us to obtain a set of distinct yet comparable samples, each associated with a different likelihood value. For each model, we evaluate the likelihood of the generated samples and compare that with AO-ARM-E's estimate since it achieves the best likelihood on test data. We repeat this controlled experiment on a random set of test images. The mean Spearman's and Pearson correlation are reported to measure the strength of correlation in marginal inference likelihoods between the given model and AO-ARM-E. MAM achieves close to 4 order of magnitude speed-up in marginal inference while at comparable quality to that from AO-ARM-S. PCs are also very fast in marginal inference but there remains a gap in terms of quality. Generated samples and additional marginal inference on partial images are in Appendix C.

Molecular sets (MOSES) We test generative modeling of MAM on a benchmarking molecular dataset [46] refined from the ZINC database [61]. Same metrics are reported as Binary-MNIST. Likelihood quality is measured similarly but on random groups of test molecules instead of generated ones. The generated molecules from MAM and AO-ARM are comparable to standard state-of-the-art molecular generative models, such as CharRNN [56], JTN-VAE [26], and LatentGAN [48] (see Appendix C), with additional controllability and flexibility in any-order generation. MAM supports

Table 2: Performance Comparison on Molecular Sets  

<table><tr><td>Model</td><td>NLL (bpd) ↓</td><td>Spearman&#x27;s ↑</td><td>Pearson ↑</td><td>Marg. inf. time (s) ↓</td></tr><tr><td>AO-ARM-E-Transformer</td><td>0.652</td><td>1.0</td><td>1.0</td><td>96.87±0.04</td></tr><tr><td>AO-ARM-S-Transformer</td><td>0.655</td><td>0.996</td><td>0.994</td><td>19.32±0.01</td></tr><tr><td>MAM-Transformer</td><td>0.655</td><td>0.998</td><td>0.995</td><td>0.006±0.00</td></tr></table>

Table 3: Performance Comparison on text8  

<table><tr><td>Model</td><td>NLL (bpc) ↓</td><td>Spearman&#x27;s ↑</td><td>Pearson ↑</td><td>Marg. inf. time (s) ↓</td></tr><tr><td>Discrete Flow (8 flows)</td><td>1.23</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AO-ARM-E-Transformer</td><td>1.494</td><td>1.0</td><td>1.0</td><td>207.60 ± 0.33</td></tr><tr><td>AO-ARM-S-Transformer</td><td>1.529</td><td>0.982</td><td>0.987</td><td>41.40 ± 0.01</td></tr><tr><td>MAM-Transformer</td><td>1.529</td><td>0.937</td><td>0.945</td><td>0.005 ± 0.000</td></tr></table>

Table 4: Performance Comparison on Ising model  $\left( {{10} \times  {10}}\right)$  
Table 5: Performance Comparison on Target Lipophilicity  

<table><tr><td>Model</td><td>NLL (bpd) ↓</td><td>KL divergence ↓</td><td>Marg. inf. time (s) ↓</td></tr><tr><td>ARM-Forward-Order-MLP</td><td>0.79</td><td>-78.63</td><td>5.29±0.07e-01</td></tr><tr><td>ARM-MC-Forward-Order-MLP</td><td>24.84</td><td>-18.01</td><td>5.30±0.07e-01</td></tr><tr><td>GFlowNet-Learned-Order-MLP</td><td>0.78</td><td>-78.17</td><td>-</td></tr><tr><td>MAM-Any-Order-MLP</td><td>0.80</td><td>-77.77</td><td>3.75±0.08e-04</td></tr></table>

<table><tr><td rowspan="2">Model Distribution</td><td colspan="4">KL divergence ↓</td></tr><tr><td>logP = 4, τ = 1.0</td><td>logP = -4, τ = 1.0</td><td>logP = 4, τ = 0.1</td><td>logP = 4, τ = 0.1</td></tr><tr><td>ARM-FO-MLP</td><td>-174.25</td><td>-168.62</td><td>-167.83</td><td>-160.2</td></tr><tr><td>MAM-AO-MLP</td><td>-173.07</td><td>-166.43</td><td>-165.75</td><td>-157.59</td></tr></table>

much faster marginal inference, which is useful for domain scientists to reason about likelihood of (sub)structures. Generated molecules and property histogram plots of are available in Appendix C.

Text8 Text8 [37] is a widely used character level natural language modeling dataset. The dataset comprises of 100M characters from Wikipedia, split into chunks of 250 character. We follow the same testing procedure as Binary-MNIST and report the same metrics. The test NLL of discrete flow is from [65], for which there are no open-source implementations to evaluate additional metrics.

# 6.2 ENERGY-BASED TRAINING

We compare with ARM that uses sum of conditionals to evaluate  $\log p_{\phi}$  with fixed forward ordering and ARM-MC that uses a one-step conditional to estimate  $\log p_{\phi}$ . ARM can be regarded as the golden standard of learning autoregressive conditionals, since its gradient needs to be evaluated on the full generation trajectory, which is the most informative and costly. MAM uses marginal network to evaluate  $\log p_{\theta}$  and subsamples a one-step marginalization constraint for each data point in the batch. The effective batch size for ARM and GFlowNet is  $B \times \mathcal{O}(D)$  for batch of size  $B$ , and  $B \times \mathcal{O}(1)$  for ARM-MC and MAM. MAM and ARM optimizes KL divergence using REINFORCE gradient estimator with baseline. GFlowNet is trained on per-sample gradient of squared distance [75].

Ising model Ising models [24] model interacting spins and are widely studied in mathematics and physics (see MacKay [35]). We study Ising model on a square lattice. The spins of the  $D$  sites are represented a  $D$ -dimensional binary vector and its distribution is  $p^{*}(\mathbf{x}) \propto f^{*}(\mathbf{x}) = \exp(-\mathcal{E}_{J}(\mathbf{x}))$  where  $\mathcal{E}_{\mathbf{J}}(\mathbf{x}) \triangleq -\mathbf{x}^{\top}\mathbf{J}\mathbf{x} - \boldsymbol{\theta}^{\top}\mathbf{x}$ , with  $\mathbf{J}$  the binary adjacency matrix. These models, although simplistic, bear analogies to the complex behavior of high-entropy alloys [9]. We compare MAM with ARM, ARM-MC, and GFlowNet on a  $10 \times 10$  ( $D = 100$ ) and a larger  $30 \times 30$  ( $D = 900$ ) Ising model where ARMs and GFlowNets fail to scale. 2000 ground truth samples are generated following Grathwohl et al. [17] and we measure test negative log-likelihood on those samples. We also measure  $D_{\mathrm{KL}}(p_{\theta}(\mathbf{x})||p^{*})$  by sampling from the learned model and evaluating  $\sum_{i=1}^{M}(\log p_{\theta}(\mathbf{x}_i) - \log f^{*}(\mathbf{x}_i))$ . Figure 5 contains KDE plots of  $-\mathcal{E}_{\mathbf{J}}(\mathbf{x})$  for the generated samples. As described in Section 4.3, the ARM-MC gradient suffers from high variance and fails to converge. It also tends to collapse and converge to a single sample. MAM has significant speedup in marginal inference and is the only model that supports any-order generative modeling. The performance in terms of KL divergence and likelihood are only slightly worse than models with fixed/learned order, which is expected since any-order modeling is harder than fixed-order modeling, and MAM is solving a more complicated task

![](images/6f74c5334d1b1892f76b07b8c4020df45910bac773f957c93c8dde8bd08d8f57.jpg)  
Figure 5: Ising model: 2000 samples are generated for each method.

![](images/185e65235f29e61da1767683886db0c94c5ac660cd166c9e548807256ee44d39.jpg)

![](images/98a7e33b84ad626c39e98c3783128e1bb944f9b40da96bcfc0b3963c6ac4756d.jpg)  
Figure 6: Target property matching: 2000 samples are generated for each method.

![](images/5b442e6d192ad927b46d4e3938c49e0affbbf06320f6d83a832ce944c97d4938.jpg)

![](images/56d48672ad46c491078b5ee11f642765509273a8cce0cf9ba73bae59a6d3542a.jpg)  
Figure 7: Conditionally generate towards low lipophilicity from a user-defined substructure in any given order. Left: Masking out the left 4 SELFIES characters. Right: Masking the right 4-20 SELFIES characters.

of jointly learning conditionals and marginals. On a  $30 \times 30$  ( $D = 900$ ) Ising model, MAM achieves a bpd of 0.835 on ground-truth samples while ARM and GFlowNet fails to scale. Distribution of generated samples is shown in Figure 5.

Molecular generation with target property In this task, we are interested in training generative models towards a specific target property of interest  $g(x)$ , such as lipophilicity (logP), synthetic accessibility (SA) etc. We define the distribution of molecules to follow  $p^* (x)\propto \exp (-(g(x) - g^*)^2 /\tau)$  where  $g^{*}$  is the target value of the property and  $\tau$  is a temperature parameter. We train ARM and MAM for lipophilicity of target values 4.0 and -4.0, both with  $\tau = 1.0$  and  $\tau = 0.1$ . Both models are trained for 4000 iterations with batch size 512. Results are shown in Figure 6 and Table 5 (additional figures in Appendix C). Findings are consistent with the Ising model experiments. Again, MAM performs just marginally below ARM. However, only MAM supports any-order modeling and scales to high-dimensional problems. Figure 6 (right) shows molecular generation with MAM for  $D = 500$ .

# 7 CONCLUSION

In conclusion, marginalization models are a novel family of generative models for high-dimensional discrete data that offer scalable and flexible generative modeling with tractable likelihoods. These models explicitly model all induced marginal distributions, allowing for fast evaluation of arbitrary marginal probabilities with a single forward pass of the neural network. MAMs also support scalable training objectives for any-order generative modeling, which previous methods struggle to achieve under the energy-based training setting. Potential future work includes designing new neural network architectures that automatically satisfy the marginalization self-consistency.

# REFERENCES

[1] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981-17993, 2021. (page 16)  
[2] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. Advances in Neural Information Processing Systems, 34:27381-27394, 2021. (pages 6 and 15)  
[3] Samy Bengio and Yoshua Bengio. Taking on the curse of dimensionality in joint distributions using neural networks. IEEE Transactions on Neural Networks, 11(3):550-557, 2000. (pages 1 and 3)  
[4] Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, and Emmanuel Bengio. Gflownet foundations. Journal of Machine Learning Research, 24(210):1-55, 2023. (pages 6 and 15)  
[5] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015. (page 20)  
[6] Y Choi, Antonio Vergari, and Guy Van den Broeck. Probabilistic circuits: A unifying framework for tractable probabilistic models. UCLA. URL: http://starai.cs.ucla.edu/papers/ProbCirc20.pdf, 2020. (pages 6 and 16)  
[7] CKCN Chow and Cong Liu. Approximating discrete probability distributions with dependence trees. IEEE transactions on Information Theory, 14(3):462-467, 1968. (page 6)  
[8] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989. (page 14)  
[9] James Damewood, Daniel Schwalbe-Koda, and Rafael Gómez-Bombarelli. Sampling lattices in semi-grand canonical ensemble with autoregressive machine learning. npj Computational Materials, 8(1):61, 2022. (pages 3, 6, and 8)  
[10] Adnan Darwiche. A differential approach to inference in Bayesian networks. Journal of the ACM (JACM), 50(3):280-305, 2003. (page 6)  
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. (page 1)  
[12] Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014. (page 17)  
[13] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. arXiv preprint arXiv:1605.08803, 2016. (page 17)  
[14] Laura Douglas, Iliyan Zarov, Konstantinos Gourgoulias, Chris Lucas, Chris Hart, Adam Baker, Maneesh Sahani, Yura Perov, and Saurabh Johri. A universal marginalizer for amortized inference in generative models. Advances in Approximate Bayesian Inference, NIPS 2017 Workshop, 2017. (page 6)  
[15] Daniel Flam-Shepherd, Kevin Zhu, and Alán Aspuru-Guzik. Language models can learn complex molecular distributions. Nature Communications, 13(1):3293, 2022. (page 3)  
[16] Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. Made: Masked autoencoder for distribution estimation. In International conference on machine learning, pp. 881-889. PMLR, 2015. (page 6)  
[17] Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, and Chris Maddison. Oops I took a gradient: Scalable sampling for discrete distributions. In International Conference on Machine Learning, pp. 3831-3841. PMLR, 2021. (pages 8 and 20)  
[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020. (page 2)  
[19] Emiel Hoogeboom, Jorn Peters, Rianne Van Den Berg, and Max Welling. Integer discrete flows and lossless compression. Advances in Neural Information Processing Systems, 32, 2019. (page 17)

[20] Emiel Hoogeboom, Alexey A Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. arXiv preprint arXiv:2110.02037, 2021. (pages 1, 3, 5, 6, 14, 16, 21, and 22)  
[21] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forre, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454-12465, 2021. (page 16)  
[22] Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks, 4(2):251-257, 1991. (page 14)  
[23] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989. (page 14)  
[24] Ernst Ising. Beitrag zur Theorie des Ferromagnetismus. Zeitschrift fur Physik, 31(1):253-258, February 1925. doi: 10.1007/BF02980577. (page 8)  
[25] Oleg Ivanov, Michael Figurnov, and Dmitry Vetrov. Variational autoencoder with arbitrary conditioning. arXiv preprint arXiv:1806.02382, 2018. (page 6)  
[26] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In International conference on machine learning, pp. 2323-2332. PMLR, 2018. (page 7)  
[27] Daniel D Johnson, Jacob Austin, Rianne van den Berg, and Daniel Tarlow. Beyond in-place corruption: Insertion and deletion in denoising probabilistic models. arXiv preprint arXiv:2107.07675, 2021. (page 16)  
[28] Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. Advances in Neural Information Processing Systems, 29, 2016. (page 17)  
[29] Mario Krenn, Florian Häse, AkshitKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik. Self-referencing embedded strings (SELFIES): A  $100\%$  robust molecular string representation. Machine Learning: Science and Technology, 1(4):045024, 2020. (pages 3 and 20)  
[30] Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 29-37. JMLR Workshop and Conference Proceedings, 2011. (pages 1, 3, and 6)  
[31] Yang Li, Shoaib Akbar, and Junier Oliva. Acflow: Flow models for arbitrary conditional likelihoods. In International Conference on Machine Learning, pp. 5831-5841. PMLR, 2020. (page 6)  
[32] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123-1130, 2023. (page 3)  
[33] Anji Liu, Honghua Zhang, and Guy Van den Broeck. Scaling up probabilistic circuits by latent variable distillation. arXiv preprint arXiv:2210.04398, 2022. (pages 6 and 16)  
[34] Xuejie Liu, Anji Liu, Guy Van den Broeck, and Yitao Liang. Understanding the distillation process from deep generative models to tractable probabilistic circuits. In International Conference on Machine Learning, pp. 21825-21838. PMLR, 2023. (pages 6 and 16)  
[35] David JC MacKay. Information Theory, Inference and Learning Algorithms. Cambridge University Press, 2003. (page 8)  
[36] Ali Madani, Ben Krause, Eric R Greene, Subu Subramanian, Benjamin P Mohr, James M Holton, Jose Luis Olmos Jr, Caiming Xiong, Zachary Z Sun, Richard Socher, et al. Large language models generate functional protein sequences across diverse families. Nature Biotechnology, pp. 1-8, 2023. (page 3)  
[37] Matt Mahoney. Large text compression benchmark, 2011. (page 8)  
[38] Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in gflows. Advances in Neural Information Processing Systems, 35:5955-5967, 2022. (pages 6 and 15)  
[39] Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai Zhang, and Yoshua Bengio. Gflownets and variational inference. arXiv preprint arXiv:2210.00580, 2022. (pages 6 and 16)

[40] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. Detectgpt: Zero-shot machine-generated text detection using probability curvature. arXiv preprint arXiv:2301.11305, 2023. (page 1)  
[41] Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457), 2019. (page 3)  
[42] OpenAI. ChatGPT, 2023. URL https://openai.com. (page 2)  
[43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744, 2022. (page 2)  
[44] George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. Advances in Neural Information Processing Systems, 30, 2017. (page 17)  
[45] Robert Peharz, Steven Lang, Antonio Vergari, Karl Stelzner, Alejandro Molina, Martin Trapp, Guy Van den Broeck, Kristian Kersting, and Zoubin Ghahramani. Einsum networks: Fast and scalable learning of tractable probabilistic circuits. In International Conference on Machine Learning, pp. 7563-7574. PMLR, 2020. (pages 6, 7, and 16)  
[46] Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, Sergey Golovanov, Oktai Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, et al. Molecular sets (moses): a benchmarking platform for molecular generation models. Frontiers in Pharmacology, 11:565644, 2020. (pages 7 and 23)  
[47] Hoifung Poon and Pedro M. Domingos. Sum-product networks: A new deep architecture. In Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence, 2011. (pages 6 and 7)  
[48] Oleksii Prykhodko, Simon Viet Johansson, Panagiotis-Christos Kotsias, Josep Arus-Pous, Esben Jannik Bjerrum, Ola Engkvist, and Hongming Chen. A de novo molecular generation method using latent vector based generative adversarial network. Journal of Cheminformatics, 11(1):1-13, 2019. (page 7)  
[49] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. (page 2)  
[50] Jie Ren, Peter J Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan. Likelihood ratios for out-of-distribution detection. Advances in neural information processing systems, 32, 2019. (page 1)  
[51] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pp. 1530–1538. PMLR, 2015. (page 17)  
[52] Oren Rippel and Ryan P. Adams. High-dimensional probability estimation with deep density models. arXiv preprint arXiv:1302.5125, 2013. (page 17)  
[53] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention-MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234-241. Springer, 2015. (pages 1 and 3)  
[54] Ruslan Salakhutdinov and Iain Murray. On the quantitative analysis of deep belief networks. In Proceedings of the 25th international conference on Machine learning, pp. 872-879, 2008. (page 20)  
[55] Arne Schneuing, Yuanqi Du, Charles Harris, Arian Jamasb, Ilia Igashov, Weitao Du, Tom Blundell, Pietro Lio, Carla Gomes, Max Welling, et al. Structure-based drug design with equivariant diffusion models. arXiv preprint arXiv:2210.13695, 2022. (page 1)  
[56] Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS Central Science, 4 (1):120-131, 2018. (pages 3 and 7)  
[57] Andy Shih, Dorsa Sadigh, and Stefano Ermon. Training and inference on any-order autoregressive models the right way. arXiv preprint arXiv:2205.13554, 2022. (pages 6, 16, and 22)

[58] Jung-Eun Shin, Adam J Riesselman, Aaron W Kollasch, Conor McMahon, Elana Simon, Chris Sander, Aashish Manglik, Andrew C Kruse, and Debora S Marks. Protein design and variant prediction using autoregressive generative models. Nature Communications, 12(1):2403, 2021. (page 3)  
[59] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pp. 2256–2265. PMLR, 2015. (pages 2, 16, and 17)  
[60] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. (page 2)  
[61] Teague Sterling and John J Irwin. ZINC 15-ligand discovery for everyone. Journal of Chemical Information and Modeling, 55(11):2324-2337, 2015. (page 7)  
[62] Ryan Strauss and Junier B Oliva. Arbitrary conditional distributions with energy. Advances in Neural Information Processing Systems, 34:752-763, 2021. (page 6)  
[63] Esteban G Tabak and Cristina V Turner. A family of nonparametric density estimation algorithms. Communications on Pure and Applied Mathematics, 66(2):145-164, 2013. (page 17)  
[64] Tijmen Tieleman. Training restricted Boltzmann machines using approximations to the likelihood gradient. In Proceedings of the 25th International Conference on Machine Learning, pp. 1064-1071, 2008. (page 5)  
[65] Dustin Tran, Keyon Vafa, Kumar Agrawal, Laurent Dinh, and Ben Poole. Discrete flows: Invertible generative models of discrete data. Advances in Neural Information Processing Systems, 32, 2019. (pages 6, 8, and 17)  
[66] Benigno Uria, Iain Murray, and Hugo Larochelle. A deep and tractable density estimator. In International Conference on Machine Learning, pp. 467-475. PMLR, 2014. (pages 1, 3, 5, 6, and 14)  
[67] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pp. 1747-1756. PMLR, 2016. (page 6)  
[68] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017. (pages 1 and 3)  
[69] Jue Wang, Sidney Lisanza, David Juergens, Doug Tischer, Joseph L Watson, Karla M Castro, Robert Ragotte, Amijai Saragovi, Lukas F Milles, Minkyung Baek, et al. Scaffolding protein functional sites using deep learning. Science, 377(6604):387-394, 2022. (page 1)  
[70] David Weininger, Arthur Weininger, and Joseph L Weininger. SMILES. 2. algorithm for generation of unique SMILES notation. Journal of Chemical Information and Computer Sciences, 29(2):97-101, 1989. (pages 3 and 20)  
[71] Dian Wu, Lei Wang, and Pan Zhang. Solving statistical mechanics using variational autoregressive networks. Physical review letters, 122(8):080602, 2019. (pages 3 and 6)  
[72] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. XLnet: Generalized autoregressive pretraining for language understanding. Advances in Neural Information Processing Systems, 32, 2019. (pages 1 and 6)  
[73] Raymond A Yeh, Chen Chen, Teck Yian Lim, Alexander G Schwing, Mark Hasegawa-Johnson, and Minh N Do. Semantic image inpainting with deep generative models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5485-5493, 2017. (page 1)  
[74] Dinghuai Zhang, Ricky TQ Chen, Nikolay Malkin, and Yoshua Bengio. Unifying generative models with gflownets. arXiv preprint arXiv:2209.02606, 2022. (pages 6 and 15)  
[75] Dinghuai Zhang, Nikolay Malkin, Zhen Liu, Alexandra Volokhova, Aaron Courville, and Yoshua Bengio. Generative flow networks for discrete probabilistic modeling. In International Conference on Machine Learning, pp. 26412-26428. PMLR, 2022. (pages 6, 7, 8, and 15)
