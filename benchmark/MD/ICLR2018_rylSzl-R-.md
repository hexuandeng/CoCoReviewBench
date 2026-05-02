# ON UNIFYING DEEP GENERATIVE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep generative models have achieved impressive success in recent years. Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), as powerful frameworks for deep generative model learning, have largely been considered as two distinct paradigms and received extensive independent studies respectively. This paper aims to establish formal connections between GANs and VAEs through a new formulation of them. We interpret sample generation in GANs as performing posterior inference, and show that GANs and VAEs involve minimizing KL divergences of respective posterior and inference distributions with opposite directions, extending the two learning phases of classic wake-sleep algorithm, respectively. The unified view provides a powerful tool to analyze a diverse set of existing model variants, and enables to transfer techniques across research lines in a principled way. For example, we apply the importance weighting method in VAE literatures for improved GAN learning, and enhance VAEs with an adversarial mechanism that leverages generated samples. Experiments show generality and effectiveness of the transferred techniques.

# 1 INTRODUCTION

Deep generative models define distributions over a set of variables organized in multiple layers. Early forms of such models dated back to works on hierarchical Bayesian models (Neal, 1992) and neural network models such as Helmholtz machines (Dayan et al., 1995), originally studied in the context of unsupervised learning, latent space modeling, etc. Such models are usually trained via an EM style framework, using either a variational inference (Jordan et al., 1999) or a data augmentation (Tanner & Wong, 1987) algorithm. Of particular relevance to this paper is the classic wake-sleep algorithm dates by Hinton et al. (1995) for training Helmholtz machines, as it explored an idea of minimizing a pair of KL divergences in opposite directions of the posterior and its approximation.

In recent years there has been a resurgence of interests in deep generative modeling. The emerging approaches, including Variational Autoencoders (VAEs) (Kingma & Welling, 2013), Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), Generative Moment Matching Networks (GMMNs) (Li et al., 2015; Dziugaite et al., 2015), auto-regressive neural networks (Larochelle & Murray, 2011; Oord et al., 2016), and so forth, have led to impressive results in a myriad of applications, such as image and text generation (Radford et al., 2015; Hu et al., 2017; van den Oord et al., 2016), disentangled representation learning (Chen et al., 2016; Kulkarni et al., 2015), and semi-supervised learning (Salimans et al., 2016; Kingma et al., 2014).

The deep generative model literature has largely viewed these approaches as distinct model training paradigms. For instance, GANs aim to achieve an equilibrium between a generator and a discriminator; while VAEs are devoted to maximizing a variational lower bound of the data log-likelihood. A rich array of theoretical analyses and model extensions have been developed independently for GANs (Arjovsky & Bottou, 2017; Arora et al., 2017; Salimans et al., 2016; Nowozin et al., 2016) and VAEs (Burda et al., 2015; Chen et al., 2017; Hu et al., 2017), respectively. A few works attempt to combine the two objectives in a single model for improved inference and sample generation (Larsen et al., 2015; Makhzani et al., 2015; Sønderby et al., 2017; Tran et al., 2017). Despite the significant progress specific to each method, it remains unclear how these apparently divergent approaches connect to each other in a principled way.

In this paper, we present a new formulation of GANs and VAEs that connects them under a unified view, and links them back to the classic wake-sleep algorithm. We show that GANs and VAEs

involve minimizing opposite KL divergences of respective posterior and inference distributions, and extending the sleep and wake phases, respectively, for generative model learning. More specifically, we develop a reformulation of GANs that interprets generation of samples as performing posterior inference, leading to an objective that resembles variational inference as in VAEs. As a counterpart, VAEs in our interpretation contain a degenerated adversarial mechanism that blocks out generated samples and only allows real examples for model training.

The proposed interpretation provides a useful tool to analyze the broad class of recent GAN- and VAE-based algorithms, enabling perhaps a more principled and unified view of the landscape of generative modeling. For instance, one can easily extend our formulation to subsume InfoGAN (Chen et al., 2016) that additionally infers hidden representations of examples, VAE/GAN joint models (Larsen et al., 2015; Che et al., 2017a) that offer improved generation and reduced mode missing, and adversarial domain adaptation (ADA) (Ganin et al., 2016; Purushotham et al., 2017) that is traditionally framed in the discriminative setting.

The close parallelisms between GANs and VAEs further ease transferring techniques that were originally developed for improving each individual class of models, to in turn benefit the other class. We provide two examples in such spirit: 1) Drawn inspiration from importance weighted VAE (IWAE) (Burda et al., 2015), we straightforwardly derive importance weighted GAN (IWGAN) that maximizes a tighter lower bound on the marginal likelihood compared to the vanilla GAN. 2) Motivated by the GAN adversarial game we activate the originally degenerated discriminator in VAEs, resulting in a full-fledged model that adaptively leverages both real and fake examples for learning. Empirical results show that the techniques imported from the other class are generally applicable to the base model and its variants, yielding consistently better performance.

# 2 RELATED WORK

There has been a surge of research interest in deep generative models in recent years, with remarkable progress made in understanding several class of algorithms. The wake-sleep algorithm (Hinton et al., 1995) is one of the earliest general approaches for learning deep generative models. The algorithm incorporates a separate inference model for posterior approximation, and aims at maximizing a variational lower bound of the data log-likelihood, or equivalently, minimizing the KL divergence of the approximate posterior and true posterior. However, besides the wake phase that minimizes the KL divergence w.r.t the generative model, the sleep phase is introduced for tractability that minimizes instead the reversed KL divergence w.r.t the inference model. Recent approaches such as NVIL (Mnih & Gregor, 2014) and VAEs (Kingma & Welling, 2013) are developed to maximize the variational lower bound w.r.t both the generative and inference models jointly. To reduce the variance of stochastic gradient estimates, VAEs leverage reparametrized gradients. Many works have been done along the line of improving VAEs. Burda et al. (2015) develop importance weighted VAEs to obtain a tighter lower bound. As VAEs do not involve a sleep phase-like procedure, generated samples from the generative model are not leveraged for model learning. Hu et al. (2017) combine VAEs with an extended sleep procedure that exploits generated samples for learning.

Another emerging family of deep generative models is the Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), in which a discriminator is trained to distinguish between real and generated samples and the generator to confuse the discriminator. The adversarial approach can be alternatively motivated in the perspectives of approximate Bayesian computation (Gutmann et al., 2014) and density ratio estimation (Mohamed & Lakshminarayanan, 2016). The original objective of the generator is to minimize the log probability of the discriminator correctly recognizing a generated sample as fake. This is equivalent to minimizing a lower bound on the Jensen-Shannon divergence (JSD) of the generator and data distributions (Goodfellow et al., 2014; Nowozin et al., 2016; Huszar, 2016; Li, 2016). Besides, the objective suffers from vanishing gradient with strong discriminator. Thus in practice people have used another objective which maximizes the log probability of the discriminator recognizing a generated sample as real (Goodfellow et al., 2014; Arjovsky & Bottou, 2017). The second objective has the same optimal solution as with the original one. We base our analysis of GANs on the second objective as it is widely used in practice yet few theoretic analysis has been done on it. Numerous extensions of GANs have been developed, including combination with VAEs for improved generation (Larsen et al., 2015; Makhzani et al., 2015; Che et al., 2017a), and generalization of the objectives to minimize other f-divergence criteria beyond JSD (Nowozin

et al., 2016; Sønderby et al., 2017). The adversarial principle has gone beyond the generation setting and been applied to other contexts such as domain adaptation (Ganin et al., 2016; Purushotham et al., 2017), and Bayesian inference (Mescheder et al., 2017; Tran et al., 2017; Huszár, 2017; Rosca et al., 2017) which uses implicit variational distributions in VAEs and leverage the adversarial approach for optimization. This paper starts from the basic models of GANs and VAEs, and develops a general formulation that reveals underlying connections of different classes of approaches including many of the above variants, yielding a unified view of the broad set of deep generative modeling.

# 3 BRIDGING THE GAP

The structures of GANs and VAEs are at the first glance quite different from each other. VAEs are based on the variational inference approach, and include an explicit inference model that reverses the generative process defined by the generative model. On the contrary, in traditional view GANs lack an inference model, but instead have a discriminator that judges generated samples. In this paper, a key idea to bridge the gap is to interpret the generation of samples in GANs as performing inference, and the discrimination as a generative process that produces real/fake labels. The resulting new formulation reveals the connections of GANs to traditional variational inference. The reversed generation-inference interpretations between GANs and VAEs also expose their correspondence to the two learning phases in the classic wake-sleep algorithm.

For ease of presentation and to establish a systematic notation for the paper, we start with a new interpretation of Adversarial Domain Adaptation (ADA) (Ganin et al., 2016), the application of adversarial approach in the domain adaptation context. We then show GANs are a special case of ADA, followed with a series of analysis linking GANs, VAEs, and their variants in our formulation.

# 3.1 ADVERSARIAL DOMAIN ADAPTATION (ADA)

ADA aims to transfer prediction knowledge learned from a source domain to a target domain, by learning domain-invariant features (Ganin et al., 2016). That is, it learns a feature extractor whose output cannot be distinguished by a discriminator between the source and target domains.

We first review the conventional formulation of ADA. Figure 1(a) illustrates the computation flow. Let  $z$  be a data example either in the source or target domain, and  $y \in \{0,1\}$  the domain indicator with  $y = 0$  indicating the target domain and  $y = 1$  the source domain. The data distributions conditioning on the domain are then denoted as  $p(z|y)$ . The feature extractor  $G_{\theta}$  parameterized with  $\theta$  maps  $z$  to feature  $x = G_{\theta}(z)$ . To enforce domain invariance of feature  $x$ , a discriminator  $D_{\phi}$  is learned. Specifically,  $D_{\phi}(x)$  outputs the probability that  $x$  comes from the source domain, and the discriminator is trained to maximize the binary classification accuracy of recognizing the domains:

$$
\max  _ {\boldsymbol {\phi}} \mathcal {L} _ {\boldsymbol {\phi}} = \mathbb {E} _ {\boldsymbol {x} = G _ {\theta} (\boldsymbol {z}), \boldsymbol {z} \sim p (\boldsymbol {z} | y = 1)} \left[ \log D _ {\boldsymbol {\phi}} (\boldsymbol {x}) \right] + \mathbb {E} _ {\boldsymbol {x} = G _ {\theta} (\boldsymbol {z}), \boldsymbol {z} \sim p (\boldsymbol {z} | y = 0)} \left[ \log \left(1 - D _ {\boldsymbol {\phi}} (\boldsymbol {x})\right) \right]. \tag {1}
$$

The feature extractor  $G_{\theta}$  is then trained to fool the discriminator:

$$
\max  _ {\boldsymbol {\theta}} \mathcal {L} _ {\boldsymbol {\theta}} = \mathbb {E} _ {\boldsymbol {x} = G _ {\boldsymbol {\theta}} (\boldsymbol {z}), \boldsymbol {z} \sim p (\boldsymbol {z} | y = 1)} \left[ \log \left(1 - D _ {\phi} (\boldsymbol {x})\right) \right] + \mathbb {E} _ {\boldsymbol {x} = G _ {\boldsymbol {\theta}} (\boldsymbol {z}), \boldsymbol {z} \sim p (\boldsymbol {z} | y = 0)} \left[ \log D _ {\phi} (\boldsymbol {x}) \right]. \tag {2}
$$

Please see the supplementary materials for more details of ADA.

With the background of conventional formulation, we now frame our new interpretation of ADA. The data distribution  $p(z|y)$  and deterministic transformation  $G_{\theta}$  together form an implicit distribution over  $\mathbf{x}$ , denoted as  $p_{\theta}(\mathbf{x}|y)$ , which is intractable to evaluate likelihood but easy to sample from. Let  $p(y)$  be the distribution of the domain indicator  $y$ , e.g., a uniform distribution as in Eqs.(1)-(2). The discriminator defines a conditional distribution  $q_{\phi}(y|\mathbf{x}) = D_{\phi}(\mathbf{x})$ . Let  $q_{\phi}^{r}(y|\mathbf{x}) = q_{\phi}(1 - y|\mathbf{x})$  be the reversed distribution over domains. The objectives of ADA are therefore rewritten as (omitting the constant scale factor 2):

$$
\max  _ {\phi} \mathcal {L} _ {\phi} = \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} [ \log q _ {\phi} (\boldsymbol {y} | \boldsymbol {x}) ]
$$

$$
\begin{array}{l} \max  _ {\boldsymbol {\phi}} \mathcal {L} _ {\boldsymbol {\phi}} = \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} [ \log q _ {\boldsymbol {\phi}} ^ {r} (y | \boldsymbol {x}) ] \\ \max  _ {\boldsymbol {\theta}} \mathcal {L} _ {\boldsymbol {\theta}} = \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} \left[ \log q _ {\boldsymbol {\phi}} ^ {r} (y | \boldsymbol {x}) \right]. \end{array} \tag {3}
$$

Note that  $\mathbf{z}$  is encapsulated in the implicit distribution  $p_{\theta}(\mathbf{x}|y)$ . The only difference of the objectives of  $\pmb{\theta}$  from  $\phi$  is the replacement of  $q(y|\mathbf{x})$  with  $q^{r}(y|\mathbf{x})$ . This is where the adversarial mechanism comes about. We defer deeper interpretation of the new objectives in the next subsection.

![](images/0074673969c88e254d50513258e473eb72e771c09839e768f8ee7d3a049938fa.jpg)  
(a)

![](images/82b77becb977fe4a0ccfba1a66eeac7706677e35b19db9eabb23f78497eae44c.jpg)  
(b)  
Figure 1: (a) Conventional view of ADA. To make direct correspondence to GANs, we use  $z$  to denote the data and  $x$  the feature. Subscripts src and tgt denote source and target domains, respectively. (b) Conventional view of GANs. (c) Schematic graphical model of both ADA and GANs (Eq.3). Arrows with solid lines denote generative process; arrows with dashed lines denote inference; hollow arrows denote deterministic transformation leading to implicit distributions; and blue arrows denote adversarial mechanism that involves respective conditional distribution  $q$  and its reverse  $q^r$ , e.g.,  $q(y|x)$  and  $q^r(y|x)$  (denoted as  $q^{(r)}(y|x)$  for short). Note that in GANs we have interpreted  $x$  as latent variable and  $(z, y)$  as visible. (d) InfoGAN (Eq.9), which, compared to GANs, adds conditional generation of code  $z$  with distribution  $q_{\eta}(z|x, y)$ . (e) VAEs (Eq.12), which is obtained by swapping the generation and inference processes of InfoGAN, i.e., in terms of the schematic graphical model, swapping solid-line arrows (generative process) and dashed-line arrows (inference) of (d).

![](images/cf41b7d368d327ea34eda13a58c2a67edb208027d72c193de6c0afbf6922a1bb.jpg)  
(c)

![](images/30a09c305ee0e2bc4ade935b92fd40e341fe06ff55da569a63a180ddc9042003.jpg)  
(d)

![](images/a3ec0fe64d64d00e330f555d199d37ac0de90ff6bbb27ef2be0a1d2d449574b7.jpg)  
(e)

# 3.2 GENERATIVE ADVERSARIAL NETWORKS (GANS)

GANs (Goodfellow et al., 2014) can be seen as a special case of ADA. Taking image generation for example, intuitively, we want to transfer the properties of real image (source domain) to generated image (target domain), making them indistinguishable to the discriminator. Figure 1(b) shows the conventional view of GANs.

Formally,  $\pmb{x}$  now denotes a real example or a generated sample,  $\pmb{z}$  is the respective latent code. For the generated sample domain  $(y = 0)$ , the implicit distribution  $p_{\theta}(\pmb{x}|y = 0)$  is defined by the prior of  $\pmb{z}$  and the generator  $G_{\theta}(\pmb{z})$ , which is also denoted as  $p_{g_{\theta}}(\pmb{x})$  in the literature. For the real example domain  $(y = 1)$ , the code space and generator are degenerated, and we are directly presented with a fixed distribution  $p(\pmb{x}|y = 1)$ , which is just the real data distribution  $p_{data}(\pmb{x})$ . Note that  $p_{data}(\pmb{x})$  is also an implicit distribution and allows efficient empirical sampling. In summary, the conditional distribution over  $\pmb{x}$  is constructed as

$$
p _ {\theta} (\boldsymbol {x} | y) = \left\{ \begin{array}{l l} p _ {g _ {\theta}} (\boldsymbol {x}) & y = 0 \\ p _ {d a t a} (\boldsymbol {x}) & y = 1. \end{array} \right. \tag {4}
$$

Here, free parameters  $\theta$  are only associated with  $p_{g_\theta}(\boldsymbol{x})$  of the generated sample domain, while  $p_{data}(\boldsymbol{x})$  is constant. As in ADA, discriminator  $D_{\phi}$  is simultaneously trained to infer the probability that  $\boldsymbol{x}$  comes from the real data domain. That is,  $q_{\phi}(y = 1|\boldsymbol{x}) = D_{\phi}(\boldsymbol{x})$ .

With the established correspondence between GANs and ADA, we can see that the objectives of GANs are precisely expressed as Eq.(3). To make this clearer, we recover the classical form by unfolding over  $y$  and plugging in conventional notations. For instance, the objective of the generative parameters  $\theta$  in Eq.(3) is translated into

$$
\begin{array}{l} \max  _ {\boldsymbol {\theta}} \mathcal {L} _ {\boldsymbol {\theta}} = \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y = 0) p (y = 0)} \left[ \log q _ {\phi} ^ {r} (y = 0 | \boldsymbol {x}) \right] + \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y = 1) p (y = 1)} \left[ \log q _ {\phi} ^ {r} (y = 1 | \boldsymbol {x}) \right] \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {x} = G _ {\theta} (\boldsymbol {z}), \boldsymbol {z} \sim p (\boldsymbol {z} | y = 0)} \left[ \log D _ {\phi} (\boldsymbol {x}) \right] + c o n s t, \tag {5} \\ \end{array}
$$

where  $p(y)$  is uniform and results in the constant scale factor  $1/2$ . As noted in sec.2, we focus on the unsaturated objective for the generator (Goodfellow et al., 2014), as it is commonly used in practice yet still lacks systematic analysis.

New Interpretation Let us take a closer look into the form of Eq.(3). It closely resembles the data reconstruction term of a variational lower bound by treating  $y$  as visible variable while  $\pmb{x}$  as latent (as in ADA). That is, we are essentially reconstructing the real/fake indicator  $y$  (or its reverse  $1 - y$ ) with the "generative distribution"  $q_{\phi}(y|\pmb{x})$  and conditioning on  $\pmb{x}$  from the "inference distribution"  $p_{\theta}(\pmb{x}|\pmb{y})$ . Figure 1(c) shows a schematic graphical model that illustrates such generative and inference processes. (Sec.D in the supplementary materials gives an example of translating a given schematic graphical model into mathematical formula.) We go a step further to reformulate the objectives and reveal more insights to the problem. In particular, for each optimization step of  $p_{\theta}(\pmb{x}|\pmb{y})$  at point  $(\pmb{\theta}_0,\pmb{\phi}_0)$  in the parameter space, we have:

![](images/f75fca8cd9cd331093ec4c00e45039f82784e36f332c39894d47dd10a4a8f6f7.jpg)  
Figure 2: One optimization step of the parameter  $\theta$  through Eq.(6) at point  $\theta_0$ . The posterior  $q^r (\pmb {x}|y)$  is a mixture of  $p_{\theta_0}(\pmb {x}|y = 0)$  (blue) and  $p_{\theta_0}(\pmb {x}|y = 1)$  (red in the left panel) with the mixing weights induced from  $q_{\phi_0}^r (y|\pmb {x})$ . Minimizing the KLD drives  $p_{\theta}(\pmb {x}|y = 0)$  towards the respective mixture  $q^{r}(\pmb {x}|y = 0)$  (green), resulting in a new state where  $p_{\theta^{new}}(\pmb {x}|y = 0) = p_{g_{\theta^{new}}}\left(\pmb {x}\right)$  (red in the right panel) gets closer to  $p_{\theta_0}(\pmb {x}|y = 1) = p_{data}(\pmb {x})$ . Due to the asymmetry of KLD,  $p_{g_{\theta^{new}}}\left(\pmb {x}\right)$  missed the smaller mode of the mixture  $q^{r}(\pmb {x}|y = 0)$  which is a mode of  $p_{data}(\pmb {x})$ .

Lemma 1. Let  $p(y)$  be the uniform distribution. Let  $p_{\theta_0}(\pmb{x}) = \mathbb{E}_{p(y)}[p_{\theta_0}(\pmb{x}|\mathcal{Y})]$ , and  $q^r (\pmb{x}|\mathcal{Y}) \propto q_{\phi_0}^r (y|\pmb{x})p_{\theta_0}(\pmb{x})$ . Therefore, the updates of  $\pmb{\theta}$  at  $\pmb{\theta}_0$  have

$$
\left. \nabla_ {\theta} \left[ - \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} \left[ \log q _ {\phi_ {0}} ^ {r} (y | \boldsymbol {x}) \right] \right] \right| _ {\boldsymbol {\theta} = \boldsymbol {\theta} _ {0}} = \tag {6}
$$

$$
\nabla_ {\theta} \left[ \right. \mathbb {E} _ {p (y)} \left[ \right. K L \left( \right.p _ {\theta} (\boldsymbol {x} | y) \left\| \right. q ^ {r} (\boldsymbol {x} | y)\left. \right)\left. \right] - J S D \left( \right.p _ {\theta} (\boldsymbol {x} | y = 0) \left\| \right. p _ {\theta} (\boldsymbol {x} | y = 1)\left. \right)\left. \right]\left. \right| _ {\boldsymbol {\theta} = \boldsymbol {\theta} _ {0}},
$$

where  $KL(\cdot \|\cdot)$  and  $JSD(\cdot \|\cdot)$  are the KL and Jensen-Shannon Divergences, respectively.

Proofs are in the supplements. Eq.(6) offers several insights into the GAN generator learning:

- Resemblance to variational inference. As above, we see  $\mathbf{x}$  as latent and  $p_{\theta}(\mathbf{x}|y)$  as the inference distribution. The  $p_{\theta_0}(\mathbf{x})$  is fixed to the starting state of the current update step, and can naturally be seen as the prior over  $\mathbf{x}$ . By definition  $q^{r}(\mathbf{x}|y)$  that combines the prior  $p_{\theta_0}(\mathbf{x})$  and the generative distribution  $q_{\phi_0}^r (y|\mathbf{x})$  thus serves as the posterior. Therefore, optimizing the generator  $G_{\theta}$  is equivalent to minimizing the KL divergence between the inference distribution and the posterior (a standard from of variational inference), minus a JSD between the distributions  $p_{g_\theta}(\mathbf{x})$  and  $p_{data}(\mathbf{x})$ . The interpretation further reveals the connections to VAEs, as discussed later.  
- Training dynamics. By definition,  $p_{\theta_0}(\pmb{x}) = (p_{g_{\theta_0}}(\pmb{x}) + p_{data}(\pmb{x})) / 2$  is a mixture of  $p_{g_{\theta_0}}(\pmb{x})$  and  $p_{data}(\pmb{x})$  with uniform mixing weights, so the posterior  $q^r(\pmb{x}|y) \propto q_{\phi_0}^r(y|\pmb{x}) p_{\theta_0}(\pmb{x})$  is also a mixture of  $p_{g_{\theta_0}}(\pmb{x})$  and  $p_{data}(\pmb{x})$  with mixing weights induced from the discriminator  $q_{\phi_0}^r(y|\pmb{x})$ . For the KL divergence to minimize, the component with  $y = 1$  is KL  $(p_{\theta}(\pmb{x}|y = 1) \| q^r(\pmb{x}|y = 1)) = \mathrm{KL}(p_{data}(\pmb{x}) \| q^r(\pmb{x}|y = 1))$  which is a constant. The active component for optimization is with  $y = 0$ , i.e., KL  $(p_{\theta}(\pmb{x}|y = 0) \| q^r(\pmb{x}|y = 0)) = \mathrm{KL}(p_{g_{\theta}}(\pmb{x}) \| q^r(\pmb{x}|y = 0))$ . Thus, minimizing the KL divergence in effect drives  $p_{g_{\theta}}(\pmb{x})$  to a mixture of  $p_{g_{\theta_0}}(\pmb{x})$  and  $p_{data}(\pmb{x})$ . Since  $p_{data}(\pmb{x})$  is fixed,  $p_{g_{\theta}}(\pmb{x})$  gets closer to  $p_{data}(\pmb{x})$ . Figure 2 illustrates the training dynamics schematically.  
- Explanation of missing mode issue. The negative JSD term is due to the introduction of the prior  $p_{\theta_0}(\pmb{x})$ . As JSD is symmetric, the missing mode behavior widely observed in GANs (Metz et al., 2017; Che et al., 2017a) is explained by the asymmetry of the KLD which tends to concentrate  $p_{\theta}(\pmb{x}|y)$  to large modes of  $q^{r}(\pmb{x}|y)$  and ignore smaller ones. See Figure 2 for the illustration. Concentration to few large modes also facilitates GANs to generate sharp and realistic samples.  
- Optimality assumption of the discriminator. Previous theoretical works have typically assumed (near) optimal discriminator (Goodfellow et al., 2014; Arjovsky & Bottou, 2017):

$$
q _ {\phi_ {0}} (y | \boldsymbol {x}) \approx \frac {p _ {\theta_ {0}} (\boldsymbol {x} | y = 1)}{p _ {\theta_ {0}} (\boldsymbol {x} | y = 0) + p _ {\theta_ {0}} (\boldsymbol {x} | y = 1)} = \frac {p _ {d a t a} (\boldsymbol {x})}{p _ {g _ {\theta_ {0}}} (\boldsymbol {x}) + p _ {d a t a} (\boldsymbol {x})}, \tag {7}
$$

which can be unwarranted in practice due to limited expressiveness of the discriminator (Arora et al., 2017). In contrast, our result does not rely on the optimality assumptions. Indeed, our result is a generalization of the previous theorem in (Arjovsky & Bottou, 2017), which is recovered by plugging Eq.(7) into Eq.(6):

$$
\left. \nabla_ {\theta} \left[ - \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} \left[ \log q _ {\phi_ {0}} ^ {r} (y | \boldsymbol {x}) \right] \right] \right| _ {\boldsymbol {\theta} = \boldsymbol {\theta} _ {0}} = \nabla_ {\theta} \left[ \frac {1}{2} \mathrm {K L} \left(p _ {g _ {\theta}} \| p _ {d a t a}\right) - \mathrm {J S D} \left(p _ {g _ {\theta}} \| p _ {d a t a}\right) \right] \Bigg | _ {\boldsymbol {\theta} = \boldsymbol {\theta} _ {0}}, \tag {8}
$$

which gives simplified explanations of the training dynamics and the missing mode issue only when the discriminator meets certain optimality criteria. Our generalized result enables understanding of broader situations. For instance, when the discriminator distribution  $q_{\phi_0}(y|x)$  gives uniform guesses, or when  $p_{g_\theta} = p_{data}$  that is indistinguishable by the discriminator, the gradients of the KL and JSD terms in Eq.(6) cancel out, which stops the generator learning.

InfoGAN Chen et al. (2016) developed InfoGAN which additionally recovers (part of) the latent code  $\mathbf{z}$  given sample  $\mathbf{x}$ . This can straightforwardly be formulated in our framework by introducing an extra conditional  $q_{\eta}(\mathbf{z}|\mathbf{x},y)$  parameterized by  $\pmb{\eta}$ . As discussed above, GANs assume a degenerated code space for real examples, thus  $q_{\eta}(\mathbf{z}|\mathbf{x},y = 1)$  is fixed without free parameters to learn, and  $\pmb{\eta}$  is only associated to  $y = 0$ . The InfoGAN is then recovered by combining  $q_{\eta}(\mathbf{z}|\mathbf{x},y)$  with  $q_{\phi}(y|\mathbf{x})$  in Eq.(3) to perform full reconstruction of both  $\mathbf{z}$  and  $y$ :

$$
\max  _ {\phi} \mathcal {L} _ {\phi} = \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} [ \log q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {\phi} (y | \boldsymbol {x}) ]
$$

$$
\left. \max  _ {\theta , \eta} \mathcal {L} _ {\theta , \eta} = \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} \left[ \log q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {\phi} ^ {r} (y | \boldsymbol {x}) \right]. \right. \tag {9}
$$

Again, note that  $\mathbf{z}$  is encapsulated in the implicit distribution  $p_{\theta}(\mathbf{x}|y)$ . The model is expressed as the schematic graphical model in Figure 1(d). Let  $q^{r}(\mathbf{x}|\mathbf{z},y)\propto q_{\eta_{0}}(\mathbf{z}|\mathbf{x},y)q_{\phi_{0}}^{r}(y|\mathbf{x})p_{\theta_{0}}(\mathbf{x})$  be the augmented "posterior", the result in the form of Lemma.1 still holds by adding  $\mathbf{z}$ -related conditionals:

$$
\left. \nabla_ {\theta} \left[ - \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | y) p (y)} \left[ \log q _ {\eta_ {0}} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {\phi_ {0}} ^ {r} (y | \boldsymbol {x}) \right] \right] \right| _ {\boldsymbol {\theta} = \boldsymbol {\theta} _ {0}} =
$$

$$
\left. \nabla_ {\theta} \left[ \mathbb {E} _ {p (y)} \left[ \mathrm {K L} \left(p _ {\theta} (\boldsymbol {x} | y) \| q ^ {r} (\boldsymbol {x} | z, y)\right) \right] - \mathrm {J S D} \left(p _ {\theta} (\boldsymbol {x} | y = 0) \| p _ {\theta} (\boldsymbol {x} | y = 1)\right) \right] \right| _ {\boldsymbol {\theta} = \boldsymbol {\theta} _ {0}}, \tag {10}
$$

The new formulation is also generally applicable to other GAN-related variants, such as Adversarial Autoencoder (Makhzani et al., 2015), Predictability Minimization (Schmidhuber, 1992), and cycleGAN (Zhu et al., 2017). In the supplements we provide interpretations of the above models.

# 3.3 VARIATIONAL AUTOENCODERS (VAES)

We next explore the second family of deep generative modeling. The resemblance of GAN generator learning to variational inference (Lemma.1) suggests strong relations between VAEs (Kingma & Welling, 2013) and GANs. We build correspondence between them, and show that VAEs involve minimizing a KLD in an opposite direction, with a degenerated adversarial discriminator.

The conventional definition of VAEs is written as:

$$
\left. \right. \max  _ {\boldsymbol {\theta}, \boldsymbol {\eta}} \mathcal {L} _ {\boldsymbol {\theta}, \boldsymbol {\eta}} ^ {\mathrm {v a e}} = \mathbb {E} _ {p _ {d a t a} (\boldsymbol {x})} \left[ \mathbb {E} _ {\tilde {q} _ {\boldsymbol {\eta}} (\boldsymbol {z} | \boldsymbol {x})} \left[ \log \tilde {p} _ {\boldsymbol {\theta}} (\boldsymbol {x} | \boldsymbol {z}) \right] - \operatorname {K L} \left(\tilde {q} _ {\boldsymbol {\eta}} (\boldsymbol {z} | \boldsymbol {x}) \| \tilde {p} (\boldsymbol {z})\right)\right], \tag {11}
$$

where  $\tilde{p}_{\theta}(\boldsymbol{x}|\boldsymbol{z})$  is the generator,  $\tilde{q}_{\eta}(\boldsymbol{z}|\boldsymbol{x})$  the inference model, and  $\tilde{p}(\boldsymbol{z})$  the prior. The parameters to learn are intentionally denoted with the notations of corresponding modules in GANs. VAEs appear to differ from GANs greatly as they use only real examples and lack adversarial mechanism.

To connect to GANs, we assume a perfect discriminator  $q_{*}(y|\boldsymbol{x})$  which always predicts  $y = 1$  with probability 1 given real examples, and  $y = 0$  given generated samples. Again, for notational simplicity, let  $q_{*}^{r}(y|\boldsymbol{x}) = q_{*}(1 - y|\boldsymbol{x})$  be the reversed distribution.

Lemma 2. Let  $p_{\theta}(z,y|\boldsymbol {x})\propto p_{\theta}(\boldsymbol {x}|\boldsymbol {z},y)p(\boldsymbol {z}|\boldsymbol {y})p(\boldsymbol {y})$  . The VAE objective  $\mathcal{L}_{\theta ,\eta}^{\mathrm{vae}}$  in Eq.(11) is equivalent to (omitting the constant scale factor 2):

$$
\begin{array}{l} \mathcal {L} _ {\theta , \eta} ^ {v a e} = \mathbb {E} _ {p _ {\theta_ {0}} (\boldsymbol {x})} \left[ \mathbb {E} _ {q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {*} ^ {r} (y | \boldsymbol {x})} \left[ \log p _ {\theta} (\boldsymbol {x} | \boldsymbol {z}, y) \right] - K L \left(q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {*} ^ {r} (y | \boldsymbol {x}) \| p (\boldsymbol {z} | y) p (y)\right) \right] \\ = \mathbb {E} _ {p _ {\theta_ {0}} (\boldsymbol {x})} \left[ - K L \left(q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {*} ^ {r} (y | \boldsymbol {x}) \| p _ {\theta} (\boldsymbol {z}, y | \boldsymbol {x})\right) \right]. \\ \end{array}
$$

Here most of the components have exact correspondences (and the same definitions) in GANs and InfoGAN (see Table 1), except that the generation distribution  $p_{\theta}(\pmb{x}|\pmb{z},y)$  differs slightly from its counterpart  $p_{\theta}(\pmb{x}|y)$  in Eq.(4) to additionally account for the uncertainty of generating  $\pmb{x}$  given  $\pmb{z}$ :

$$
p _ {\theta} (\boldsymbol {x} | \boldsymbol {z}, y) = \left\{ \begin{array}{l l} \tilde {p} _ {\theta} (\boldsymbol {x} | \boldsymbol {z}) & y = 0 \\ p _ {\text {d a t a}} (\boldsymbol {x}) & y = 1. \end{array} \right. \tag {13}
$$

We provide the proof of Lemma 2 in the supplementary materials. Figure 1(e) shows the schematic graphical model of the new interpretation of VAEs, where the only difference from InfoGAN (Figure 1(d)) is swapping the solid-line arrows (generative process) and dashed-line arrows (inference). As in GANs and InfoGAN, for the real example domain with  $y = 1$ , both  $q_{\eta}(\pmb{z}|\pmb{x},y = 1)$  and  $p_{\theta}(\pmb{x}|\pmb{z},y = 1)$  are constant distributions. Since given a fake sample  $\pmb{x}$  from  $p_{\theta_0}(\pmb{x})$ , the reversed perfect discriminator  $q_{*}^{r}(y|\pmb{x})$  always predicts  $y = 1$  with probability 1, the loss on fake samples is therefore degenerated to a constant, which blocks out fake samples from contributing to learning.

<table><tr><td>Components</td><td>ADA</td><td>GANs / InfoGAN</td><td>VAEs</td></tr><tr><td>x</td><td>features</td><td>data/generations</td><td>data/generations</td></tr><tr><td>y</td><td>domain indicator</td><td>real/fake indicator</td><td>real/fake indicator (degenerated)</td></tr><tr><td>z</td><td>data examples</td><td>code vector</td><td>code vector</td></tr><tr><td>pθ(x|y)</td><td>feature distr.</td><td>[I] generator, Eq.4</td><td>[G] pθ(x|z,y), generator, Eq.13</td></tr><tr><td>qφ(y|x)</td><td>discriminator</td><td>[G] discriminator</td><td>[I] q*(y|x), discriminator (degenerated)</td></tr><tr><td>qη(z|x,y)</td><td>—</td><td>[G] infer net (InfoGAN)</td><td>[I] infer net</td></tr><tr><td>KLD to min</td><td>same as GANs</td><td>KL (pθ(x|y)||qr(x|y))</td><td>KL (qη(z|x,y)q*(y|x)||pθ(z,y|x))</td></tr></table>

Table 1: Correspondence between different approaches in the proposed formulation. The label "[G]" in bold indicates the respective component is involved in the generative process within our interpretation, while "[I]" indicates inference process. This is also expressed in the schematic graphical models in Figure 1.

# 3.4 CONNECTING GANS AND VAES

Table 1 summarizes the correspondence between the approaches. Lemma.1 and Lemma.2 have revealed that both GANs and VAEs involve minimizing a KLD of respective inference and posterior distributions. In particular, GANs involve minimizing the  $KL\left(p_{\theta}(\boldsymbol{x}|y)\big||q^{r}(\boldsymbol{x}|y)\right)$  while VAEs the  $KL\left(q_{\eta}(\boldsymbol{z}|\boldsymbol{x},y)q_{*}^{r}(\boldsymbol{y}|\boldsymbol{x})\big||p_{\theta}(\boldsymbol{z},y|\boldsymbol{x})\right)$ . This exposes several new connections between the two model classes, each of which in turn leads to a set of existing research, or can inspire new research directions:

1) As discussed in Lemma.1, GANs now also relate to the variational inference algorithm as with VAEs, revealing a unified statistical view of the two classes. Moreover, the new perspective naturally enables many of the extensions of VAEs and vanilla variational inference algorithm to be transferred to GANs. We show an example in the next section.  
2) The generator parameters  $\theta$  are placed in the opposite directions in the two KLDs. The asymmetry of KLD leads to distinct model behaviors. For instance, as discussed in Lemma.1, GANs are able to generate sharp images but tend to collapse to one or few modes of the data (i.e., mode missing). In contrast, the KLD of VAEs tends to drive generator to cover all modes of the data distribution but also small-density regions (i.e., mode covering), which usually results in blurred, implausible samples. This naturally inspires combination of the two KLD objectives to remedy the asymmetry. Previous works have explored such combinations, though motivated in different perspectives (Larsen et al., 2015; Che et al., 2017a). We discuss more details in the supplements.  
3) VAEs within our formulation also include adversarial mechanism as in GANs. The discriminator is perfect and degenerated, disabling generated samples to help with learning. This inspires activating the adversary to allow learning from samples. We present a simple possible way in the next section.  
4) GANs and VAEs have inverted latent-visible treatments of  $(z, y)$  and  $x$ , since we interpret sample generation in GANs as posterior inference. Such inverted treatments strongly relates to the symmetry of the sleep and wake phases in the wake-sleep algorithm, as presented shortly. In sec.A of the supplementary materials, we provide a more general discussion on a symmetric view of generation and inference.

# 3.5 CONNECTING TO WAKE SLEEP ALGORITHM (WS)

Wake-sleep algorithm (Hinton et al., 1995) was proposed for learning deep generative models such as Helmholtz machines (Dayan et al., 1995). WS consists of wake phase and sleep phase, which optimize the generative model and inference model, respectively. We follow the above notations, and introduce new notations  $h$  to denote general latent variables and  $\lambda$  to denote general parameters. The wake sleep algorithm is thus written as:

$$
\text {W a k e}: \quad \max  _ {\theta} \mathbb {E} _ {q _ {\lambda} (\boldsymbol {h} | \boldsymbol {x}) p _ {d a t a} (\boldsymbol {x})} [ \log p _ {\theta} (\boldsymbol {x} | \boldsymbol {h}) ] \tag {14}
$$

$$
\text {S l e e p}: \quad \max  _ {\boldsymbol {\lambda}} \mathbb {E} _ {p _ {\theta} (\boldsymbol {x} | \boldsymbol {h}) p (\boldsymbol {h})} \left[ \log q _ {\boldsymbol {\lambda}} (\boldsymbol {h} | \boldsymbol {x}) \right].
$$

Briefly, the wake phase updates the generator parameters  $\pmb{\theta}$  by fitting  $p_{\theta}(\pmb{x}|\pmb{h})$  to the real data and hidden code inferred by the inference model  $q_{\lambda}(\pmb{h}|\pmb{x})$ . On the other hand, the sleep phase updates the parameters  $\lambda$  based on the generated samples from the generator.

The relations between WS and VAEs are clear in previous discussions (Bornschein & Bengio, 2014; Kingma & Welling, 2013). Indeed, WS was originally proposed to minimize the variational lower bound as in VAEs (Eq.11) with the sleep phase approximation (Hinton et al., 1995). Alternatively,

VAEs can be seen as extending the wake phase. Specifically, if we let  $h$  be  $z$  and  $\lambda$  be  $\eta$ , the wake phase objective recovers VAEs (Eq.11) in terms of generator optimization (i.e., optimizing  $\theta$ ). Therefore, we can see VAEs as generalizing the wake phase by also optimizing the inference model  $q_{\eta}$ , with additional prior regularization on code  $z$ .

On the other hand, GANs closely resemble the sleep phase. To make this clearer, let  $h$  be  $y$  and  $\lambda$  be  $\phi$ . This results in a sleep phase objective identical to that of optimizing the discriminator  $q_{\phi}$  in Eq.(3), which is to reconstruct  $y$  given sample  $x$ . We thus can view GANs as generalizing the sleep phase by also optimizing the generative model  $p_{\theta}$  to reconstruct reversed  $y$ . InfoGAN (Eq.9) further extends the correspondence to reconstruction of latents  $z$ .

# 4 TRANSFERRING TECHNIQUES

The new interpretation not only reveals the connections underlying the broad set of existing approaches, but also facilitates to exchange ideas and transfer techniques across the two classes of algorithms. For instance, existing enhancements on VAEs can straightforwardly be applied to improve GANs, and vice versa. This section gives two examples. Here we only outline the main intuitions and resulting models, while providing the details in the supplement materials.

# 4.1 IMPORTANCE WEIGHTED GANS (IWGAN)

Burda et al. (2015) proposed importance weighted autoencoder (IWAE) that maximizes a tighter lower bound on the marginal likelihood. Within our framework it is straightforward to develop importance weighted GANs by copying the derivations of IWAE side by side, with little adaptations. Specifically, the variational inference interpretation in Lemma.1 suggests GANs can be viewed as maximizing a lower bound of the marginal likelihood on  $y$  (putting aside the negative JSD term):

$$
\log q (y) = \log \int p _ {\theta} (\boldsymbol {x} | y) \frac {q _ {\phi_ {0}} ^ {r} (y | \boldsymbol {x}) p _ {\theta_ {0}} (\boldsymbol {x})}{p _ {\theta} (\boldsymbol {x} | y)} d \boldsymbol {x} \geq - \mathrm {K L} \left(p _ {\theta} (\boldsymbol {x} | y) \| q ^ {r} (\boldsymbol {x} | y)\right) + c o n s t. \tag {15}
$$

Following (Burda et al., 2015), we can derive a tighter lower bound through a  $k$ -sample importance weighting estimate of the marginal likelihood. With necessary approximations for tractability, optimizing the tighter lower bound results in the following update rule for the generator learning:

$$
\nabla_ {\theta} \mathcal {L} _ {k} (y) = \mathbb {E} _ {\boldsymbol {z} _ {1}, \dots , \boldsymbol {z} _ {k} \sim p (\boldsymbol {z} | y)} \left[ \sum_ {i = 1} ^ {k} \widetilde {w _ {i}} \nabla_ {\theta} \log q _ {\phi_ {0}} ^ {r} (y | \boldsymbol {x} (\boldsymbol {z} _ {i}, \boldsymbol {\theta})) \right]. \tag {16}
$$

As in GANs, only  $y = 0$  (i.e., generated samples) is effective for learning parameters  $\theta$ . Compared to the vanilla GAN update (Eq.(6)), the only difference here is the additional importance weight  $\widetilde{w_i}$  which is the normalization of  $w_{i} = \frac{q_{\phi_{0}}^{r}(y|\boldsymbol{x}_{i})}{q_{\phi_{0}}(y|\boldsymbol{x}_{i})}$  over  $k$  samples. Intuitively, the algorithm assigns higher weights to samples that are more realistic and fool the discriminator better, which is consistent to IWAE that emphasizes more on code states providing better reconstructions. Hjelm et al. (2017); Che et al. (2017b) developed a similar sample weighting scheme for maximum likelihood generator training. In practice, the  $k$  samples correspond to sample minibatch in standard GAN update. Thus the only computational cost added by the importance weighting method is by evaluating the weight for each sample, and is negligible. The discriminator is trained in the same way as in standard GANs.

# 4.2 ADVERSARY ACTIVATED VAES (AAVAE)

By Lemma.2, VAEs include a degenerated discriminator which blocks out generated samples from contributing to model learning. We enable adaptive incorporation of fake samples by activating the adversarial mechanism. Specifically, we replace the perfect discriminator  $q_{*}(y|\boldsymbol{x})$  in VAEs with a discriminator network  $q_{\phi}(y|\boldsymbol{x})$  parameterized with  $\phi$ , resulting in an adapted objective of Eq.(12):

$$
\max _ {\boldsymbol {\theta}, \boldsymbol {\eta}} \mathcal {L} _ {\boldsymbol {\theta}, \boldsymbol {\eta}} ^ {\mathrm {a v a v e}} = \mathbb {E} _ {p _ {\theta_ {0}} (\boldsymbol {x})} \left[ \mathbb {E} _ {q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {\phi} ^ {r} (y | \boldsymbol {x})} \left[ \log p _ {\theta} (\boldsymbol {x} | \boldsymbol {z}, y) \right] - \operatorname {K L} \left(q _ {\eta} (\boldsymbol {z} | \boldsymbol {x}, y) q _ {\phi} ^ {r} (y | \boldsymbol {x}) \| p (\boldsymbol {z} | y) p (y)\right) \right]. \tag {17}
$$

As detailed in the supplementary material, the discriminator is trained in the same way as in GANs.

The activated discriminator enables an effective data selection mechanism. First, AAVAE uses not only real examples, but also generated samples for training. Each sample is weighted by the inverted

<table><tr><td></td><td>GAN</td><td>IWGAN</td></tr><tr><td>MNIST</td><td>8.34±.03</td><td>8.45±.04</td></tr><tr><td>SVHN</td><td>5.18±.03</td><td>5.34±.03</td></tr></table>

<table><tr><td></td><td>CGAN</td><td>IWCGAN</td></tr><tr><td>MNIST</td><td>0.985±.002</td><td>0.987±.002</td></tr><tr><td>SVHN</td><td>0.797±.005</td><td>0.798±.006</td></tr></table>

<table><tr><td></td><td>SVAE</td><td>AASVAE</td></tr><tr><td>1%</td><td>0.9412</td><td>0.9425</td></tr><tr><td>10%</td><td>0.9768</td><td>0.9797</td></tr></table>

Table 2: Left: Inception scores of GANs and the importance weighted extension. Middle: Classification accuracy of the generations by conditional GANs and the IW extension. Right: Classification accuracy of semi-supervised VAEs and the AA extension on MNIST test set, with  $1\%$  and  $10\%$  real labeled training data.  

<table><tr><td>Train Data Size</td><td>VAE</td><td>AA-VAE</td><td>CVAE</td><td>AA-CVAE</td><td>SVAE</td><td>AA-SVAE</td></tr><tr><td>1%</td><td>-122.89</td><td>-122.15</td><td>-125.44</td><td>-122.88</td><td>-108.22</td><td>-107.61</td></tr><tr><td>10%</td><td>-104.49</td><td>-103.05</td><td>-102.63</td><td>-101.63</td><td>-99.44</td><td>-98.81</td></tr><tr><td>100%</td><td>-92.53</td><td>-92.42</td><td>-93.16</td><td>-92.75</td><td>—</td><td>—</td></tr></table>

Table 3: Variational lower bounds on MNIST test set, trained on  $1\%$ ,  $10\%$ , and  $100\%$  training data, respectively. In the semi-supervised VAE (SVAE) setting, remaining training data are used for unsupervised training.

discriminator  $q_{\phi}^{r}(y|\boldsymbol{x})$ , so that only those samples that resemble real data and successfully fool the discriminator will be incorporated for training. This is consistent with the importance weighting strategy in IWGAN. Second, real examples are also weighted by  $q_{\phi}^{r}(y|\boldsymbol{x})$ . An example receiving large weight indicates it is easily recognized by the discriminator, which means the example is hard to be simulated from the generator. That is, AAVAE emphasizes more on harder examples.

# 5 EXPERIMENTS

Quantitative experiments show the importance weighting (IW) and adversary activating (AA) extensions improve the standard GANs and VAEs, as well as several of their variants, respectively. We present the results here, and provide details of experimental setups in the supplements.

# 5.1 IMPORTANCE WEIGHTED GANS

We extend both vanilla GANs and class-conditional GANs (CGAN) with the IW method. The base GAN model is implemented with the DCGAN architecture and hyperparameter setting (Radford et al., 2015). Hyperparameters are not tuned for the IW extensions. We use MNIST and SVHN for evaluation. For vanilla GANs and its IW extension, we measure inception scores (Salimans et al., 2016) on the generated samples. For CGANs we evaluate the accuracy of conditional generation (Hu et al., 2017) with a pre-trained classifier. Please see the supplements for more details.

Table 2, left panel, shows the inception scores of GANs and IW-GAN, and the middle panel gives the classification accuracy of CGAN and and its IW extension. We report the averaged results  $\pm$  one standard deviation over 5 runs. The IW strategy gives consistent improvements over the base models.

# 5.2 ADVERSARY ACTIVATED VAES

We apply the AA method on vanilla VAEs, class-conditional VAEs (CVAE), and semi-supervised VAEs (SVAE) (Kingma et al., 2014), respectively. We evaluate on the MNIST data. We measure the variational lower bound on the test set, with varying number of real training examples. For each batch of real examples, AA extended models generate equal number of fake samples for training.

Table 3 shows the results of activating the adversarial mechanism in VAEs. Generally, larger improvement is obtained with smaller set of real training data. Table 2, right panel, shows the improved accuracy of AA-SVAE over the base semi-supervised VAE.

# 6 CONCLUSIONS

Our new interpretations of GANs and VAEs have revealed strong connections between them, and linked the emerging new approaches to the classic wake-sleep algorithm. The generality of the proposed formulation offers a unified statistical insight of the broad landscape of deep generative modeling, and encourages mutual exchange of techniques across research lines. One of the key ideas in our formulation is to interpret sample generation in GANs as performing posterior inference. We provide a more general discussion of this point in sec.A of the supplements. It is interesting to further generalize the framework to connect to other learning paradigms such as reinforcement learning as previous works have started exploration (Finn et al., 2016; Pfau & Vinyals, 2016).

# REFERENCES

Martin Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. In ICLR, 2017.  
Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang. Generalization and equilibrium in generative adversarial nets (GANs). arXiv preprint arXiv:1703.00573, 2017.  
Mark A Beaumont, Wenyang Zhang, and David J Balding. Approximate Bayesian computation in population genetics. Genetics, 162(4):2025-2035, 2002.  
Jörg Bornschein and Yoshua Bengio. Reweighted wake-sleep. arXiv preprint arXiv:1406.2751, 2014.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
Tong Che, Yanran Li, Athul Paul Jacob, Yoshua Bengio, and Wenjie Li. Mode regularized generative adversarial networks. *ICLR*, 2017a.  
Tong Che, Yanran Li, Ruixiang Zhang, R Devon Hjelm, Wenjie Li, Yangqiu Song, and Yoshua Bengio. Maximum-likelihood augmented discrete generative adversarial networks. arXiv preprint:1702.07983, 2017b.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. InfoGAN: Interpretable representation learning by information maximizing generative adversarial nets. In NIPS, 2016.  
Xi Chen, Diederik P Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, and Pieter Abbeel. Variational lossy autoencoder. *ICLR*, 2017.  
Peter Dayan, Geoffrey E Hinton, Radford M Neal, and Richard S Zemel. The helmholtz machine. Neural computation, 7(5):889-904, 1995.  
Gintare Karolina Dziugaite, Daniel M Roy, and Zoubin Ghahramani. Training generative neural networks via maximum mean discrepancy optimization. arXiv preprint arXiv:1505.03906, 2015.  
Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv preprint arXiv:1611.03852, 2016.  
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. JMLR, 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, pp. 2672-2680, 2014.  
Michael U Gutmann, Ritabrata Dutta, Samuel Kaski, and Jukka Corander. Statistical inference of intractable generative models via classification. arXiv preprint arXiv:1407.4981, 2014.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Geoffrey E Hinton, Peter Dayan, Brendan J Frey, and Radford M Neal. The" wake-sleep" algorithm for unsupervised neural networks. Science, 268(5214):1158, 1995.  
R Devon Hjelm, Athul Paul Jacob, Tong Che, Kyunghyun Cho, and Yoshua Bengio. Boundary-seeking generative adversarial networks. arXiv preprint arXiv:1702.08431, 2017.  
Zhiting Hu, Xuezhe Ma, Zhengzhong Liu, Eduard Hovy, and Eric Xing. Harnessing deep neural networks with logic rules. In ACL, 2016.  
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, and Eric P Xing. Toward controlled generation of text. In ICML, 2017.  
Ferenc Huszar. InfoGAN: using the variational bound on mutual information (twice). Blogpost, 2016. URL http://www.inference.vc/infogan-variational-bound-on-mutual-information-twice.  
Ferenc Huszár. Variational inference using implicit distributions. arXiv preprint arXiv:1702.08235, 2017.  
Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. Machine learning, 37(2):183-233, 1999.

Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In NIPS, pp. 3581-3589, 2014.  
Tejas D Kulkarni, William F Whitney, Pushmeet Kohli, and Josh Tenenbaum. Deep convolutional inverse graphics network. In NIPS, pp. 2539-2547, 2015.  
Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In AISTATS, 2011.  
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, and Ole Winther. Autoencoding beyond pixels using a learned similarity metric. arXiv preprint arXiv:1512.09300, 2015.  
Yingzhen Li. GANs, mutual information, and possibly algorithm selection? Blogpost, 2016. URL http://www.yingzhenli.net/home/blog/?p=421.  
Yujia Li, Kevin Swersky, and Rich Zemel. Generative moment matching networks. In ICML, 2015.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.  
Lars Mescheder, Sebastian Nowozin, and Andreas Geiger. Adversarial variational Bayes: Unifying variational autoencoders and generative adversarial networks. arXiv preprint arXiv:1701.04722, 2017.  
Luke Metz, Ben Poole, David Pfau, and Sohl-Dickstein. Unrolled generative adversarial networks. *ICLR*, 2017.  
Andriy Mnih and Karol Gregor. Neural variational inference and learning in belief networks. arXiv preprint arXiv:1402.0030, 2014.  
Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. arXiv preprint arXiv:1610.03483, 2016.  
Radford M Neal. Connectionist learning of belief networks. Artificial intelligence, 56(1):71-113, 1992.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In NIPS, pp. 271-279, 2016.  
Augustus Odena, Christopher Olah, and Jonathon Shlens. Conditional image synthesis with auxiliary classifier GANs. ICML, 2017.  
Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, 2016.  
David Pfau and Oriol Vinyals. Connecting generative adversarial networks and actor-critic methods. arXiv preprint arXiv:1610.01945, 2016.  
Sanjay Purushotham, Wilka Carvalho, Tanachat Nilanon, and Yan Liu. Variational recurrent adversarial deep domain adaptation. In ICLR, 2017.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Mihaela Rosca, Balaji Lakshminarayanan, David Warde-Farley, and Shakir Mohamed. Variational approaches for auto-encoding generative adversarial networks. arXiv preprint arXiv:1706.04987, 2017.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In NIPS, pp. 2226-2234, 2016.  
Jürgen Schmidhuber. Learning factorial codes by predictability minimization. Neural Computation, 1992.  
Casper Kaae Sønderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Huszár. Amortised MAP inference for image super-resolution. *ICLR*, 2017.  
Martin A Tanner and Wing Hung Wong. The calculation of posterior distributions by data augmentation. JASA, 82(398):528-540, 1987.  
Dustin Tran, Rajesh Ranganath, and David M Blei. Deep and hierarchical implicit models. arXiv preprint arXiv:1702.08896, 2017.  
Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelCNN decoders. In NIPS, 2016.  
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593, 2017.

![](images/cd333b6a17bb192737e390394b04558c603c466c13d0703f82720db429393147.jpg)  
Figure 3: Symmetric view of generation and inference. There is little difference of the two processes in terms of formulation: with implicit distribution modeling, both processes only need to perform simulation through black-box neural transformations between the latent and visible spaces.
