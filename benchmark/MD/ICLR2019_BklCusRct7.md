# OPTIMAL TRANSPORT MAPS FOR DISTRIBUTION PRESERVING OPERATIONS ON LATENT SPACES OF GENERATIVE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Generative models such as Variational Auto Encoders (VAEs) and Generative Adversarial Networks (GANs) are typically trained for a fixed prior distribution in the latent space, such as uniform or Gaussian. After a trained model is obtained, one can sample the Generator in various forms for exploration and understanding, such as interpolating between two samples, sampling in the vicinity of a sample or exploring differences between a pair of samples applied to a third sample. However, the latent space operations commonly used in the literature so far induce a distribution mismatch between the resulting outputs and the prior distribution the model was trained on. Previous works have attempted to reduce this mismatch with heuristic modification to the operations or by changing the latent distribution and re-training models. In this paper, we propose a framework for modifying the latent space operations such that the distribution mismatch is fully eliminated. Our approach is based on optimal transport maps, which adapt the latent space operations such that they fully match the prior distribution, while minimally modifying the original operation. Our matched operations are readily obtained for the commonly used operations and distributions and require no adjustment to the training procedure.

# 1 INTRODUCTION & RELATED WORK

Generative models such as Variational Autoencoders (VAEs) (Kingma & Welling, 2013) and Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) have emerged as popular techniques for unsupervised learning of intractable distributions. In the framework of Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), the generative model is obtained by jointly training a generator  $G$  and a discriminator  $D$  in an adversarial manner. The discriminator is trained to classify synthetic samples from real ones, whereas the generator is trained to map samples drawn from a fixed prior distribution to synthetic examples which fool the discriminator. Variational Autoencoders (VAEs) (Kingma & Welling, 2013) are also trained for a fixed prior distribution, but this is done through the loss of an Autoencoder that minimizes the variational lower bound of the data likelihood. For both VAEs and GANs, using some data  $\mathcal{X}$  we end up with a trained generator  $G$ , that is supposed to map latent samples  $z$  from the fixed prior distribution to output samples  $G(z)$  which (hopefully) have the same distribution as the data.

In order to understand and visualize the learned model  $G(z)$ , it is a common practice in the literature of generative models to explore how the output  $G(z)$  behaves under various arithmetic operations on the latent samples  $z$ . However, the operations typically used so far, such as linear interpolation (Goodfellow et al., 2014), spherical interpolation (White, 2016), vicinity sampling and vector arithmetic (Radford et al., 2015), cause a distribution mismatch between the latent prior distribution and the results of the operations. This is problematic, since the generator  $G$  was trained on a fixed prior and expects to see inputs with statistics consistent with that distribution.

To address this, we propose to use distribution matching transport maps, to obtain analogous latent space operations (e.g. interpolation, vicinity sampling) which preserve the prior distribution of the latent space, while minimally changing the original operation. In Figure 1 we showcase how our proposed technique gives an interpolation operator which avoids distribution mismatch when interpolating between samples of a uniform distribution. The points of the (red) matched trajectories

![](images/0d8408e7e5cfea9f1f8dee337146f99cd32140b63a6761bb1e1183241bf1e633.jpg)  
(a) Uniform prior: Trajectories of linear interpolation, our matched interpolation and the spherical interp. (White, 2016).

![](images/00e9f9946705e5beb44024771d30dff0f933f3e22f98517b30be2099924f68fc.jpg)

![](images/074ea8a8e89fee5e732fa8327725ba7ed92c0e0a8e6010bd277b36a046010f3b.jpg)  
(b) Uniform prior distribution.  
(d) Matched midpoint distribution (ours)  
Figure 1: We show examples of distribution mismatches induced by the previous interpolation schemes when using a uniform prior in two dimensions. Our matched interpolation avoids this with a minimal modification to the linear trajectory, traversing through the space such that all points along the path are distributed identically to the prior.

![](images/229da3e6a3b889764fb604c224ae31ceaa5a0844c8386b6dccad3b7676aa66c1.jpg)

![](images/6c35b971493df321c74731392ad41164ec90212fac106b6b713c582da4cd5de1.jpg)  
(c) Linear midpoint distribution  
(e) Spherical midpoint distribution (White, 2016)

are obtained as minimal deviations (in expectation of  $l_{1}$  distance) from the points of the (blue) linear trajectory.

# 1.1 DISTRIBUTION MISMATCH AND RELATED APPROACHES

In the literature there are dozens of papers that use sample operations to explore the learned models (Bengio et al. (2013); Goodfellow et al. (2014); Dosovitskiy et al. (2015); Reed et al. (2016); Brock et al. (2016); Reed et al. (2016) to name a few), but most of them have ignored the problem of distribution mismatch. Kingma & Welling (2013) and Makhzani et al. (2015) sidestep the problem when visualizing their models, by not performing operations on latent samples, but instead restrict the latent space to 2-d and uniformly sample the percentiles of the distribution on a 2-d grid. This way, the samples have statistics that are consistent with the prior distribution. However, this approach does not scale up to higher dimensions - whereas the latent spaces used in the literature can have hundreds of dimensions.

White (2016) experimentally observe that there is a distribution mismatch between the norm for points drawn from uniform or Gaussian distribution and points obtained with linear interpolation (SLERP), and (heuristically) propose to use a so-called spherical linear interpolation to reduce the mismatch, obtaining higher quality interpolated samples.

While SLERP has been subjectively observed to produce better looking samples than linear interpolation and is now commonly, its heuristic nature has limited it from fully replacing the linear interpolation. Furthermore, while perhaps possible it is not obvious how to generalize it to other operations, such as vicinity sampling, n-point interpolation and random walk. In Section 2 we show that for interpolation, in high dimensions SLERP tends to approximately perform distribution matching the approach taken by our framework which can explain why it works well in practice.

Kilcher et al. (2018) further analyze the (norm) distribution mismatch observed by White (2016) (in terms of KL-Divergence) for the special case of Gaussian priors, and propose an alternative prior distribution with dependent components which produces less (but still nonzero) distribution mismatch for linear interpolation, at the cost of needing to re-train and re-tune the generative models.

In contrast, we propose a framework which allows one to adapt generic operations, such that they fully preserve the original prior distribution while being faithful to the original operation. Thus the KL-Divergence between the prior and the distribution of the results from our operations is zero.

The approach works as follows: we are given a 'desired' operation, such as linear interpolation  $\pmb{y} = t\pmb{z}_1 + (1 - t)\pmb{z}_2$ ,  $t \in [0,1]$ . Since the distribution of  $\pmb{y}$  does not match the prior distribution of  $\pmb{z}$ , we search for a warping  $f: \mathbb{R}^d \to \mathbb{R}^d$ , such that  $\tilde{\pmb{y}} = f(\pmb{y})$  has the same distribution as  $\pmb{z}$ . In order to have the modification  $\tilde{\pmb{y}}$  as faithful as possible to the original operation  $\pmb{y}$ , we use optimal transform maps (Santambrogio, 2015; Villani, 2003; 2008) to find a minimal modification of  $\pmb{y}$  which recovers the prior distribution  $\pmb{z}$ .

<table><tr><td>Operation</td><td>Expression</td></tr><tr><td>2-point interpolation</td><td>y = tz1 + (1 - t)z2, t ∈ [0,1]</td></tr><tr><td>n-point interpolation</td><td>y = ∑i=1n ti zi with ∑i ti = 1</td></tr><tr><td>Vicinity sampling</td><td>yj = z1 + εuj for j = 1, ..., k</td></tr><tr><td>Analogies</td><td>y = z3 + (z2 - z1)</td></tr></table>

Table 1: Examples of interesting sample operations which need to be adapted ('matched') if we want the distribution of the result  $\mathbf{y}$  to match the prior distribution.

This is illustrated in Figure 1a, where each point  $\tilde{\pmb{y}}$  of the matched curve is obtained by warping a corresponding point  $\pmb{y}$  of the linear trajectory, while not deviating too far from the line.

# 2 FROM DISTRIBUTION MISMATCH TO OPTIMAL TRANSPORT

With implicit models such as GANs (Goodfellow et al., 2014) and VAEs (Kingma & Welling, 2013), we use the data  $\mathcal{X}$ , drawn from an unknown random variable  $x$ , to learn a generator  $G: \mathbb{R}^d \mapsto \mathbb{R}^{d'}$  with respect to a fixed prior distribution  $p_z$ , such that  $G(z)$  approximates  $x$ . Once the model is trained, we can sample from it by feeding latent samples  $z$  through  $G$ .

We now bring our attention to operations on latent samples  $z_{1}, \dots, z_{k}$  from  $p_{z}$ , i.e. mappings

$$
\kappa : \mathbb {R} ^ {d} \times \dots \times \mathbb {R} ^ {d} \rightarrow \mathbb {R} ^ {d}. \tag {1}
$$

We give a few examples of such operations in Table 1.

Since the inputs to the operations are random variables, their output  $\pmb{y} = \kappa(\pmb{z}_1, \dots, \pmb{z}_k)$  is also a random variable (commonly referred to as a statistic). While we typically perform these operations on realized (i.e. observed) samples, our analysis is done through the underlying random variable  $\pmb{y}$ . The same treatment is typically used to analyze other statistics over random variables, such as the sample mean, sample variance and test statistics.

In Table 1 we show example operations which have been commonly used in the literature. As discussed in the Introduction, such operations can provide valuable insight into how the trained generator  $G$  changes as one creates related samples  $y$  from some source samples. The most common such operation is the linear interpolation, which we can view as an operation

$$
\boldsymbol {y} _ {t} = t \boldsymbol {z} _ {1} + (1 - t) \boldsymbol {z} _ {2}, \tag {2}
$$

where  $z_{1},z_{2}$  are latent samples from the prior  $p_z$  and  $y_{t}$  is parameterized by  $t\in [0,1]$ .

Now, assume  $\mathbf{z}_1$  and  $\mathbf{z}_2$  are i.i.d, and let  $Z_1, Z_2$  be their (scalar) first components with distribution  $p_Z$ . Then the first component of  $\mathbf{y}_t$  is  $Y_t = tZ_1 + (1 - t)Z_2$ , and we can compute:

$$
\operatorname {V a r} \left[ Y _ {t} \right] = \operatorname {V a r} \left[ t Z _ {1} + (1 - t) Z _ {2} \right] = t ^ {2} \operatorname {V a r} \left[ Z _ {1} \right] + (1 - t) ^ {2} \operatorname {V a r} \left[ Z _ {2} \right] = (1 + 2 t (t - 1)) \operatorname {V a r} [ Z ]. \tag {3}
$$

Since  $(1 + 2t(t - 1))\neq 1$  for all  $t\in [0,1]\setminus \{0,1\}$ , it is in general impossible for  $\pmb{y}_t$  to have the same distribution as  $\pmb{z}$ , which means that distribution mismatch is inevitable when using linear interpolation. A similar analysis reveals the same for all of the operations in Table 1.

This leaves us with a dilemma: we have various intuitive operations (see Table 1) which we would want to be able to perform on samples, but their resulting distribution  $p_{\mathbf{y}_t}$  is inconsistent with the distribution  $p_z$  we trained  $G$  for.

Due to the curse of dimensionality, as empirically observed by White (2016), this mismatch can be significant in high dimensions. We illustrate this in Figure 2, where we plot the distribution of the squared norm  $\| \pmb{y}_t\|^2$  for the midpoint  $t = 1/2$  of linear interpolation, compared to the prior distribution  $\| z\|^2$ . With  $d = 100$  (a typical dimensionality for the latent space), the distributions are dramatically different, having almost no common support. Kilcher et al. (2018) quantify this mismatch for Gaussian priors in terms of KL-Divergence, and show that it grows linearly with the dimension  $d$ . In Appendix A (see Supplement) we expand this analysis and show that this happens for all prior distributions with i.i.d. entries (i.e. not only Gaussian), both in terms of geometry and KL-Divergence.

# 2.1 DISTRIBUTION MATCHING WITH OPTIMAL TRANSPORT

In order to address the distribution mismatch, we propose a simple and intuitive framework for constructing distribution preserving operators, via optimal transport:

![](images/e7409da2e465e83f238edc987e3d054aa04042bbda0dd03bf33d57e791ad159f.jpg)  
(a) Uniform distribution

![](images/629257c3a9c77e3a2e9457453248fbb2cfdbc1af5496793fc92678e3de8238f4.jpg)  
(b) Gaussian distribution  
Figure 2: Distribution of the squared norm  $\| \pmb{y} \|^2$  of midpoints for two prior distributions in 100 dimensions: (a) components uniform on  $[-1, 1]$  and (b) components Gaussian  $\mathcal{N}(0, 1)$ , for linear interpolation, our proposed matched interpolation and the spherical interpolation proposed by White (2016). Both linear and spherical interpolation introduce a distribution mismatch, whereas our proposed matched interpolation preserves the prior distribution for both priors.

Strategy 1 (Optimal Transport Matched Operations).

1. We construct an 'intuitive' operator  $\pmb{y} = \kappa (\pmb{z}_1,\dots ,\pmb{z}_k)$  
2. We analytically (or numerically) compute the resulting (mismatched) distribution  $p_{\mathbf{y}}$  
3. We search for a minimal modification  $\tilde{\pmb{y}} = f(\pmb{y})$  (in the sense that  $E_{\pmb{y}}[c(\tilde{\pmb{y}},\pmb{y})]$  is minimal with respect to a cost  $c$ ), such that distribution is brought back to the prior, i.e.  $p_{\tilde{\pmb{y}}} = p_{z}$ .

The cost function in step 3 could e.g. be the euclidean distance  $c(x, y) = \| x - y \|$ , and is used to measure how faithful the modified operator,  $\tilde{y} = f(\kappa(z_1, \dots, z_k))$  is to the original operator  $k$ . Finding the map  $f$  which gives a minimal modification can be challenging, but fortunately it is a well studied problem from optimal transport theory. We refer to the modified operation  $\tilde{y}$  as the matched version of  $y$ , with respect to the cost  $c$  and prior distribution  $p_z$ .

For completeness, we introduce the key concept of optimal transport theory in a simplified setting, i.e. assuming probability distributions are in euclidean space and skipping measure theoretical formalism. We refer to Villani (2003; 2008) and Santambrogio (2015) for a thorough and formal treatment of optimal transport.

The problem of step (3) above was first posed by Monge (1781) and can more formally be stated as:

Problem 1 (Santambrogio (2015) Problem 1.1). Given probability distributions  $p_x, p_y$ , with domains  $\mathcal{X}, \mathcal{Y}$  respectively, and a cost function  $c: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}^+$ , we want to minimize

$$
\left. \right. \inf  \left\{ \right.E _ {\boldsymbol {x} \sim p _ {\boldsymbol {x}}} [ c (\boldsymbol {x}, f (\boldsymbol {x})) ] \left. \right| f: \mathcal {X} \rightarrow \mathcal {Y}, f (\boldsymbol {x}) \sim p _ {\boldsymbol {y}} \left. \right\} \tag {MP}
$$

We refer to the minimizer  $f^{*}\mathcal{X} \to \mathcal{Y}$  of (MP) (if it exists), as the optimal transport map from  $p_{x}$  to  $p_{y}$  with respect to the cost  $c$ .

However, the problem remained unsolved until a relaxed problem was studied by Kantorovich (1942):

Problem 2 (Santambrogio (2015) Problem 1.2). Given probability distributions  $p_x, p_y$ , with domains  $\mathcal{X}, \mathcal{Y}$  respectively, and a cost function  $c: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}^+$ , we want to minimize

$$
\left. \inf  \left\{E _ {(\boldsymbol {x}, \boldsymbol {y}) \sim p _ {\boldsymbol {x}, \boldsymbol {y}}} [ c (\boldsymbol {x}, \boldsymbol {y}) ] \right| (\boldsymbol {x}, \boldsymbol {y}) \sim p _ {\boldsymbol {x}, \boldsymbol {y}}, \boldsymbol {x} \sim p _ {\boldsymbol {x}}, \boldsymbol {y} \sim p _ {\boldsymbol {y}} \right\}, \tag {KP}
$$

where  $(\pmb{x},\pmb{y})\sim p_{\pmb{x},\pmb{y}},\pmb{x}\sim p_{\pmb{x}},\pmb{y}\sim p_{\pmb{y}}$  denotes that  $(\pmb{x},\pmb{y})$  have a joint distribution  $p_{\pmb{x},\pmb{y}}$  which has (previously specified) marginals  $p_x$  and  $p_y$ .

We refer to the joint  $p_{\pmb{x},\pmb{y}}$  which minimizes (KP) as the optimal transport plan from  $p_{\pmb{x}}$  to  $p_{\pmb{y}}$  with respect to the cost  $c$ .

The key difference is to relax the deterministic relationship between  $\pmb{x}$  and  $f(\pmb{x})$  to a joint probability distribution  $p_{\pmb{x},\pmb{y}}$  with marginals  $p_{\pmb{x}}$  and  $p_{\pmb{y}}$  for  $\pmb{x}$  and  $\pmb{y}$ . In the case of Problem 1, the minimization might be over the empty set since it is not guaranteed that there exists a mapping  $f$  such that  $f(\pmb{x}) \sim \pmb{y}$ .

In contrast, for Problem 2, one can always construct a joint density  $p_{\pmb{x},\pmb{y}}$  with  $p_{\pmb{x}}$  and  $p_{\pmb{y}}$  as marginals, such as the trivial construction where  $\pmb{x}$  and  $\pmb{y}$  are independent, i.e.  $p_{\pmb{x},\pmb{y}}(x,y) \coloneqq p_{\pmb{x}}(x)p_{\pmb{y}}(y)$ .

Note that given a joint density  $p_{\mathbf{x}, \mathbf{y}}(x, y)$  over  $\mathcal{X} \times \mathcal{Y}$ , we can view  $\mathbf{y}$  conditioned on  $\mathbf{x} = x$  for a fixed  $x$  as a stochastic function  $f(x)$  from  $\mathcal{X}$  to  $\mathcal{Y}$ , since given a fixed  $x$  do not get a specific function value  $f(x)$  but instead a random variable  $f(x)$  that depends on  $x$ , with  $f(x) \sim \mathbf{y} | \mathbf{x} = x$  with density  $p_{\mathbf{y}}(y | \mathbf{x} = x) := \frac{p_{\mathbf{x}, \mathbf{y}}(x, y)}{p_{\mathbf{x}}(x)}$ . In this case we have  $(\mathbf{x}, \mathbf{f}(\mathbf{x})) \sim p_{\mathbf{x}, \mathbf{y}}$ , so we can view the Problem KP as a relaxation of Problem MP where  $f$  is allowed to be a stochastic mapping.

While the relaxed problem of Kantorovich (KP) is much more studied in the optimal transport literature, for our purposes of constructing operators it is desirable for the mapping  $f$  to be deterministic as in (MP).

To this end, we will choose the cost function  $c$  such that the two problems coincide and where we can find an analytical solution  $f$  or at least an efficient numerical solution.

In particular, we note that the operators in Table 1 are all pointwise, such that if the points  $\mathbf{z}_i$  have i.i.d. components, then the result  $\mathbf{y}$  will also have i.i.d. components.

If we combine this with the constraint for the cost  $c$  to be additive over the components of  $x, y$ , we obtain the following simplification:

Theorem 1. Suppose  $p_x$  and  $p_y$  have i.i.d components and  $c$  over  $\mathcal{X} \times \mathcal{Y} = \mathbb{R}^d \times \mathbb{R}^d$  decomposes as

$$
c (x, y) = \sum_ {i = 1} ^ {d} C \left(x ^ {(i)}, y ^ {(i)}\right). \tag {4}
$$

Consequently, the minimization problems (MP) and (KP) turn into  $d$  identical scalar problems for the distributions  $p_X$  and  $p_Y$  of the components of  $\mathbf{x}$  and  $\mathbf{y}$ :

$$
\left. \inf  \left\{E _ {X \sim p _ {X}} [ C (X, T (X)) ] \mid T: \mathbb {R} \rightarrow \mathbb {R}, T (X) \sim p _ {Y} \right\}\right. \tag {MP-1-D}
$$

$$
\left. \inf  \left\{E _ {(X, Y) \sim p _ {X, Y}} [ C (X, Y) ] \right| (X, Y) \sim p _ {X, Y}, X \sim p _ {X}, Y \sim p _ {Y} \right\}, \tag {KP-1-D}
$$

such that an optimal transport map  $T$  for (MP-1-D) gives an optimal transport map  $f$  for (MP) by pointwise application of  $T$ , i.e.  $f(x)^{(i)} := T(x^{(i)})$ , and an optimal transport plan  $p_{X,Y}$  for (KP-1-D) gives an optimal transport plan  $p_{\boldsymbol{x},\boldsymbol{y}}(x,y) := \prod_{i=1}^{d} p_{X,Y}(x^{(i)},y^{(i)})$  for (KP).

Proof. See Appendix.

![](images/4e1e8ab4765b6fad11083b21bc370f2bf738ab7ee3eda02bf4cbce6ae0b74ef8.jpg)

Fortunately, under some mild constraints, the scalar problems have a known solution:

Theorem 2 (Theorem 2.9 in Santambrogio (2015)). Let  $h: \mathbb{R} \to \mathbb{R}^+$  be convex and suppose the cost  $C$  takes the form  $C(x, y) = h(x - y)$ . Given an continuous source distribution  $p_X$  and a target distribution  $p_Y$  on  $\mathbb{R}$  having a finite optimal transport cost in (KP-1-D), then

$$
T _ {X \rightarrow Y} ^ {m o n} (x) := F _ {Y} ^ {[ - 1 ]} \left(F _ {X} (x)\right), \tag {5}
$$

defines an optimal transport map from  $p_X$  to  $p_Y$  for (MP-1-D), where  $F_X(x) \coloneqq \int_{-\infty}^x p_X(x') dx'$  is the Cumulative Distribution Function (CDF) of  $X$  and  $F_Y^{[-1]}(y) \coloneqq \inf \{t \in \mathbb{R} | F_Y(t) \geq y\}$  is the pseudo-inverse of  $F_Y$ . Furthermore, the joint distribution of  $(X, T_{X \to Y}^{mon}(X))$  defines an optimal transport plan for (KP-1-D).

The mapping  $T_{X \to Y}^{\mathrm{mon}}(x)$  in Theorem 2 is non-decreasing and is known as the monotone transport map from  $X$  to  $Y$ . It is easy to verify that  $T_{X \to Y}^{\mathrm{mon}}(X)$  has the distribution of  $Y$ , in particular  $F_{X}(X) \sim \mathrm{Uniform}(0,1)$  and if  $U \sim \mathrm{Uniform}(0,1)$  then  $F_{Y}^{[-1]}(U) \sim Y$ .

Now, combining Theorems 1 and 2, we obtain a concrete realization of the Strategy 1 outlined above. We choose the cost  $c$  such that it admits to Theorem 1, such as  $c(\pmb{x},\pmb{y}) \coloneqq \| \pmb{x} - \pmb{y}\|_1$ , and use an operation that is pointwise, so we just need to compute the monotone transport map in (5). That is, if  $z$  has i.i.d components with distribution  $p_Z$ , we just need to compute the component distribution  $p_Y$  of the result  $y$  of the operation, the CDFs  $F_Z, F_Y$  and obtain

$$
T _ {Y \rightarrow Z} ^ {\operatorname {m o n}} (y) := F _ {Z} ^ {[ - 1 ]} \left(F _ {Y} (y)\right) \tag {6}
$$

![](images/fc97391553536013faa6d09278ca142e70011a5297d15c3e20671b9a6bf1fe87.jpg)  
(a) Uniform prior

![](images/8a73da629525cedbfab1b1a29b335f0594de1020db3a9e7a9e1fe0db0df46081.jpg)  
(b) Gaussian prior  
Figure 3: We show the monotone transport maps for linear interpolation evaluated at  $t \in \{0.05, 0.25, 0.5\}$ , to Uniform and Gaussian priors.

as the component-wise modification of  $\pmb{y}$ , i.e.  $\tilde{\pmb{y}}^{(i)}\coloneqq T_{Y\to Z}^{\mathrm{mon}}(\pmb{y}^{(i)})$

In Figure 3 we show the monotone transport map for the linear interpolation  $\pmb{y} = t\pmb{z}_1 + (1 - t)\pmb{z}_2$  for various values of  $t$ . The detailed calculations and examples for various operations are given in Appendix B, for both Uniform and Gaussian priors.

# 3 SIMULATIONS

To validate the correctness of the matched operators computed in Appendix B, we numerically simulate the distributions for toy examples, as well as prior distributions typically used in the literature.

Priors vs. interpolations in 2-D For Figure 1, we sample 1 million pairs of points in two dimension, from a uniform prior (on  $[-1,1]^2$ ), and estimate numerically the midpoint distribution of linear interpolation, our proposed matched interpolation and the spherical interpolation of White (2016). It is reassuring to see that the matched interpolation gives midpoints which are identically distributed to the prior. In contrast, the linear interpolation condenses more towards the origin, forming a pyramid-shaped distribution (the result of convolving two boxes in 2-d). Since the spherical interpolation of White (2016) follows a great circle with varying radius between the two points, we see that the resulting distribution has a "hole" in it, "circling" around the origin for both priors.

Priors vs. interpolations in 100-D For Figure 2, we sample 1 million pairs of points in  $d = 100$  dimensions, using either i.i.d. uniform components on  $[-1, 1]$  or Gaussian  $\mathcal{N}(0, 1)$  and compute the distribution of the squared norm of the midpoints. We see there is a dramatic difference between vector lengths in the prior and the midpoints of linear interpolation, with only minimal overlap. We also see that the spherical interpolation (SLERP) is approximately matching the prior (norm) distribution, having a matching first moment, but otherwise also induces a distribution mismatch. In contrast, our matched interpolation, fully preserves the prior distribution and perfectly aligns. We note that this setting ( $d = 100$ , uniform or Gaussian) is commonly used in the literature.

# 4 EXPERIMENTS

Setup We used DCGAN (Radford et al., 2015) generative models trained on LSUN bedrooms (Yu et al., 2015), CelebA (Liu et al., 2015) and LLD (Sage et al., 2017), an icon dataset, to qualitatively evaluate. For LSUN, the model was trained for two different output resolutions, providing  $64 \times 64$  pixel and a  $128 \times 128$  pixel output images (where the latter is used in figures containing larger sample images). The models for LSUN and the icon dataset were both trained on a uniform latent prior distribution, while for CelebA a Gaussian prior was used. The dimensionality of the latent space is 100 for both LSUN and CelebA, and 512 for the model trained on the icon model. Furthermore we use improved Wasserstein GAN (iWGAN) with gradient penalty (Gulrajani et al., 2017) trained on CIFAR-10 at  $32 \times 32$  pixels with a 128-dimensional Gaussian prior to compute inception scores.

# 4.1 QUANTITATIVE RESULTS

To measure the effect of the distribution mismatch, we quantitatively evaluate using the Inception score(Salimans et al., 2016). In Table 2 we compare the Inception score of our trained models (i.e.

Under review as a conference paper at ICLR 2019  

<table><tr><td>Dataset</td><td>CIFAR-10</td><td>LLD-Icon</td><td>LSUN</td><td>CelebA</td></tr><tr><td>Model</td><td>iWGAN</td><td>DCGAN</td><td>DCGAN</td><td>DCGAN</td></tr><tr><td>Prior</td><td>Gaussian, 128-D</td><td>Uniform, 100-D</td><td>Uniform, 100-D</td><td>Gaussian, 100-D</td></tr><tr><td colspan="5">Inception scores:</td></tr><tr><td>random samples</td><td>7.90 ± 0.11</td><td>3.70 ± 0.09</td><td>3.90 ± 0.08</td><td>2.05 ± 0.04</td></tr><tr><td>2-point linear</td><td>7.12 ± 0.08 (-10%)</td><td>3.56 ± 0.06 (-4%)</td><td>3.57 ± 0.07 (-8%)</td><td>1.71 ± 0.02 (-17%)</td></tr><tr><td>2-point matched</td><td>7.89 ± 0.08</td><td>3.69 ± 0.08</td><td>3.89 ± 0.08</td><td>2.04 ± 0.03</td></tr><tr><td>4-point linear</td><td>5.84 ± 0.08 (-26%)</td><td>3.45 ± 0.08 (-7%)</td><td>2.95 ± 0.06 (-24%)</td><td>1.46 ± 0.01 (-29%)</td></tr><tr><td>4-point matched</td><td>7.91 ± 0.09</td><td>3.69 ± 0.10</td><td>3.91 ± 0.10</td><td>2.04 ± 0.04</td></tr></table>

Table 2: Inception scores on LLD-icon, LSUN, CIFAR-10 and CelebA for the midpoints of linear interpolation and its matched counterpart. Scores are reported as mean ± standard deviation (relative change in %). Our matched variants fully recover from the (up to 29%) score drop of the linear interpolation, giving the same quality as random samples.

![](images/fa520ac7fe7f27be18a2df3aee8108e0e889a66c9f0ab9a39fdd4657dfb9552a.jpg)  
(a) LLD icon dataset

![](images/0fbc881e3f80763ce74ed5e55c1c106135f3176439f0321f65b81b4c749de3ad.jpg)  
(b) LSUN dataset

![](images/6598c7024b782bbdcb09738762825bb25aebcf242cded2617731d82080791d29.jpg)  
(c) CelebA dataset  
Figure 4: 2-point interpolation: Each example shows linear, SLERP and transport matched interpolation from top to bottom respectively. For LLD icon dataset (a) and LSUN (b), outputs are produced with DCGAN using a uniform prior distribution, whereas the CelebA model (c) uses a Gaussian prior. The output resolution for the (a) is  $32 \times 32$ , for (b) and (c)  $64 \times 64$  pixels.

using random samples from the prior) with the score when sampling midpoints from the 2-point and 4-point interpolations described above, reporting mean and standard deviation with 50,000 samples, as well as relative change to the original model scores if they are significant. Compared to the original scores of the trained models (random samples), our matched operations are statistically indistinguishable (as expected) while the linear interpolation gives a significantly lower score in all settings (up to  $29\%$  lower).

However, this is not surprising, since our matched operations are guaranteed to produce samples that come from the same distribution as the random samples.

![](images/7433fe621261f76bee3d6987e74a778d4e70d27534181f394adaf5e2fac16525.jpg)  
(a) Linear interpolation  
Figure 5: 4-point interpolation between 4 sampled points (corners) from DCGAN trained on LSUN  $(128\times 128)$  using a uniform prior. The same interpolation is shown using linear, SLERP and distribution matched interpolation.

![](images/0d531c94b5b65bf7b63b81e5c59c3397af6eca9a592472aaccbd13f48be1f327.jpg)  
(b) Spherical interpolation

![](images/a4821730a66e8076d5935999496af594d25d23f80944d48087536502a31690af.jpg)  
(c) Distribution matched

![](images/35920e8630dec06dabcacc97fb879fbf61c1787ab7d82f1626cbad70603d9ea3.jpg)  
Figure 6: Random walk for LLD, LSUN (64 x 64) and CelebA. The random walks consist of a succession of steps in random directions, calculated for the same sequence of directions using (non-matched) vicinity sampling in the upper rows and our proposed matched vicinity sampling in the lower rows.

# 4.2 QUALITATIVE RESULTS

In the following, we will qualitatively show that our matched operations behave as expected, and that there is a visual difference between the original operations and the matched counterparts. To this end, the generator output for latent samples produced with linear interpolation, SLERP (spherical linear interpolation) of White (2016) and our proposed matched interpolation will be compared.

2-point interpolation We begin with the classic example of 2-point interpolation: Figure 4 shows three examples per dataset for an interpolation between 2 points in latent space. Each example is first done via linear interpolation, then SLERP and finally matched interpolation. It is immediately obvious in Figures 4a and 4b that linear interpolation produces inferior results with generally more blurry, less saturated and less detailed output images.

The SLERP heuristic and matched interpolation are slightly different visually, but we do not observe a difference in visual quality. However, we stress that the goal of this work is to construct operations in a principled manner, whose samples are consistent with the generative model. In the case of linear interpolation (our framework generalizes to more operations, see below and Appendix), the SLERP heuristic tends to work well in practice but we provide a principled alternative.

4-point interpolation An even stronger effect can be observed when we do 4-point interpolation, showcased in Figure 5 (LSUN) and Figure 7 (LLD icons). The higher resolution of the LSUN output highlights the very apparent loss of detail and increasing prevalence of artifacts towards the midpoint in the linear version, compared to SLERP and our matched interpolation.

Midpoints (Appendix) In all cases, the point where the interpolation methods diverge the most, is at the midpoint of the interpolation where  $t = 0.5$ . Thus we provide 25 such interpolation midpoints in Figures 10 (LLD icons) and 11 (LSUN) in the Appendix for direct comparison.

Vicinity sampling (Appendix) Furthermore we provide two examples for vicinity sampling in Figures 8 and 9 in the Appendix. Analogous to the previous observations, the output under a linear operator lacks definition, sharpness and saturation when compared to both spherical and matched operators.

Random walk An interesting property of our matched vicinity sampling is that we can obtain a random walk in the latent space by applying it repeatedly: we start at a point  $\mathbf{y}_0 = \mathbf{z}$  drawn from the prior, and then obtain point  $\mathbf{y}_i$  by sampling a single point in the vicinity of  $\mathbf{y}_{i-1}$ , using some fixed 'step size'  $\epsilon$ . We show an example of such a walk in Figure 6, using  $\epsilon = 0.5$ . As a result of the repeated application of the vicinity sampling operation, the divergence from the prior distribution in the non-matched case becomes stronger with each step, resulting in completely unrecognizable output images on the LSUN and LLD icon models.

# 5 CONCLUSIONS

We proposed a framework that fully eliminates the distribution mismatch in the common latent space operations used for generative models. Our approach uses optimal transport to minimally modify (in  $l_{1}$  distance) the operations such that they fully preserve the prior distribution. We give analytical formulas of the resulting (matched) operations for various examples, which are easily implemented. The matched operators give a significantly higher quality samples compared to the originals, having the potential to become standard tools for evaluating and exploring generative models.

# REFERENCES

Yoshua Bengio, Grégoire Mesnil, Yann Dauphin, and Salah Rifai. Better mixing via deep representations. In Proceedings of the 30th International Conference on Machine Learning (ICML-13), pp. 552-560, 2013.  
Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Neural photo editing with introspective adversarial networks. arXiv preprint arXiv:1609.07093, 2016.  
Alexey Dosovitskiy, Jost Tobias Springenberg, and Thomas Brox. Learning to generate chairs with convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1538-1546, 2015.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans. arXiv:1704.00028v2, 2017.  
Leonid Vitalievich Kantorovich. On the translocation of masses. In Dokl. Akad. Nauk SSSR, volume 37, pp. 199-201, 1942.  
Yannic Kilcher, Aurelien Lucchi, and Thomas Hofmann. Semantic interpolation in implicit models. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=H15odZ-C-.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015.  
David JC MacKay. Information theory, inference and learning algorithms. Cambridge university press, 2003.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.  
Gaspard Monge. Mémoire sur la théorie des déblais et des remblais. Histoire de l'Académie Royale des Sciences de Paris, 1781.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. arXiv preprint arXiv:1605.05396, 2016.  
Alexander Sage, Eirikur Agustsson, Radu Timofte, and Luc Van Gool. Ltd: Large logo dataset. 2017. URL https://data.vision.ee.ethz.ch/cvl/lld/.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pp. 2234-2242, 2016.  
Filippo Santambrogio. Optimal transport for applied mathematicians. *Birkhäuser*, NY, 2015.  
Cédric Villani. Topics in optimal transportation. Number 58. American Mathematical Soc., 2003.  
Cédric Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media, 2008.  
Tom White. Sampling generative networks. arXiv preprint arXiv:1609.04468, 2016.  
Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.
