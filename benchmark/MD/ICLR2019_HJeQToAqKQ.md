# THERML:

# THE THERMODYNAMICS OF MACHINE LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this work we offer an information-theoretic framework for representation learning that connects with a wide class of existing objectives in machine learning. We develop a formal correspondence between this work and thermodynamics and discuss its implications.

# 1 INTRODUCTION

Let  $X, Y$  be some paired data, for example: a set of images  $X$  and their labels  $Y$ . We imagine the data comes from some true, unknown data generating process  $\Phi^1$ , from which we have drawn a training set of  $N$  pairs:

$$
\mathcal {T} _ {N} \equiv \left(x ^ {N}, y ^ {N}\right) \equiv \left\{x _ {1}, y _ {1}, x _ {2}, y _ {2}, \dots , x _ {N}, y _ {N} \right\} \sim \phi \left(x ^ {N}, y ^ {N}\right). \tag {1}
$$

We further imagine the process is exchangeable² and the data is conditionally independent given the governing process  $\Phi$ :

$$
p \left(x ^ {N}, y ^ {N} \mid \phi\right) = \prod_ {i} p \left(x _ {i} \mid \phi\right) p \left(y _ {i} \mid x _ {i}, \phi\right). \tag {2}
$$

As machine learners, we believe that by studying the training set, we should be able to infer or predict new draws from the same data generating process. Call a set of  $M$  future draws from the data generating process  $\mathcal{T}_M^{\prime}\equiv \{X^M,Y^M\}$  the test set.

The predictive information (Bialek et al., 2001) is the mutual information between the training set and a infinite test set, equivalently the amount of information the training set provides about the generative process itself:

$$
I _ {\text {p r e d}} \left(\mathcal {T} _ {N}\right) \equiv \lim  _ {M \rightarrow \infty} I \left(\mathcal {T} _ {N}; \mathcal {T} _ {M} ^ {\prime}\right) = I \left(\mathcal {T} _ {N}; \Phi\right) = I \left(X ^ {N}, Y ^ {N}; \Phi\right). \tag {3}
$$

The predictive information measures the underlying complexity of the data generating process (Still, 2014), and is fundamentally limited and must grow sublinearly in the dataset size (Bialek et al., 2001). Hence, the predictive information is a vanishing fraction of the total information in the training set  $^3$ :

$$
\lim  _ {N \rightarrow \infty} \frac {I _ {\operatorname* {p r e d}} \left(\mathcal {T} _ {N}\right)}{H \left(\mathcal {T} _ {N}\right)} = 0 \tag {4}
$$

A vanishing fraction of the information present in our training data is in any way useful for future tasks. A vanishing fraction of the information contained in the training data is signal, the rest is noise. We claim the goal of learning is to learn a representation of data, both locally and globally that captures the predictive information while being maximally compressed: that separates the signal from the noise.

Figure 1: Graphical models.  
![](images/1904372cb0b72fc8b93e3de5af41d4743b8e273a0ee0c19359d96e8e66375ea1.jpg)  
(a) Graphical model for world  $P$ , the real world augmented with a local and global representation. The dashed lines emphasize that  $\theta$  only depends on the first  $N$  data points, the training set. Blue denotes nodes outside our control, while red nodes are under our direct control.

![](images/352e39e1e751e8db7da359bb4238a3c040eb557871b334aa8b2831bc11bb9b2c.jpg)  
(b) Graphical model for world  $Q$ , the world we desire. In this world,  $Z$  acts as a latent variable for  $X$  and  $Y$  jointly.

# 2 A TALE OF TWO WORLDDS

We are primarily interested in learning a stochastic local representation of  $X$ , call it  $Z$ , defined by some parametric distribution of our own design:  $p(z_i|x_i,\theta)$  with its own parameters  $\theta$ . A training procedure is a process that assigns a distribution  $p(\theta |x^N,y^N)$  to the parameters conditioned on the observed dataset. In this way, the parameters of our local parametric map are themselves a global representation of the dataset. With our augmentations, the world now looks like the graphical model in Figure 3a, denoted World  $P$ : Some data generating process  $\Phi$  generates a dataset  $(X^{N},Y^{N})$  which we perform some learning algorithm on to get some parameters  $p(\theta |x^N,y^N)$  which we can use to form a parametric local representation  $p(z_i|x_i,\theta)$ .

World  $P$  is what we have. It is not necessarily what we want. What we have to contend with is an unknown distribution of our data. What we want is a world that corresponds to the traditional modeling assumptions in which  $Z$  acts as a latent factor for  $X$  and  $Y$ , rending them conditionally independent, leaving no correlations unexplained. Similarly, we would prefer if we could easily marginalize out the dependence on our universal  $(\Phi)$  and model specific  $(\Theta)$  parameters. World  $Q$  in Figure 3b is the world we want<sup>4</sup>.

We can measure the degree to which the real world aligns with our desires by computing the minimum possible relative information between our distribution  $p$  and any distribution consistent with the conditional dependencies encoded in graphical model  $Q^6$ . It can be shown (Friedman et al., 2001) that this quantity is given by the difference in multi-informations between the two graphical models, as measured in World  $P$ :

$$
\mathcal {J} \equiv \min  _ {q \in Q} D _ {\mathrm {K L}} [ p; q ] = I _ {P} - I _ {Q}. \tag {5}
$$

The multi-information (Slonim et al., 2005) of a graphical model is the KL divergence between the joint distribution and the product of all of the marginal distributions, which can be computed as a sum of mutual informations, one for each node in the graph, between itself and its parents:

$$
I _ {G} \equiv \left\langle \log \frac {p \left(g ^ {N}\right)}{\prod_ {i} p \left(g _ {i}\right)} \right\rangle = \sum_ {i} I \left(g _ {i}; \operatorname {P a} \left(g _ {i}\right)\right) \tag {6}
$$

In our case:

$$
\mathcal {J} = I (\Theta ; X ^ {N}, Y ^ {N}) + \sum_ {i} \left[ I \left(X _ {i}; \Phi\right) + I \left(Y _ {i}; X _ {i}, \Phi\right) + I \left(Z _ {i}; X _ {i}, \Theta\right) - I \left(X _ {i}; Z _ {i}\right) - I \left(Y _ {i}; Z _ {i}\right) \right]. \tag {7}
$$

This minimal relative information has two terms outside our control and we can take them to be constant, but which relate to the predictive information:

$$
\sum_ {i} \left[ I \left(X _ {i}; \Phi\right) + I \left(Y _ {i}; X _ {i}, \Phi\right) \right] \geq \sum_ {i} I \left(Y _ {i}; X _ {i}\right) + I _ {\text {p r e d}} \left(\mathcal {T} _ {N}\right). \tag {8}
$$

These terms measure the intrinsic complexity of our data. The remaining four terms are:

-  $I(X_{i};Z_{i})$  - which measures how much information our representation contains about the input  $(X)$ . This should be maximized to ensure our local representation actually represents the input.  
-  $I(Y_{i};Z_{i})$  - which measures how much information our representation contains about our auxiliary data. This should be maximized as well to ensure that our local representation is predictive for the labels.  
-  $I(Z_{i};X_{i},\Theta)$  - which measures how much information the parameters and input determine about our representation. This should be minimized to ensure consistency between worlds, and ensure we learn compressed local representations. Notice that this is similar to, but distinct from the first term above.

$$
I \left(Z _ {i}; X _ {i}, \Theta\right) = I \left(Z _ {i}; X _ {i}\right) + I \left(Z _ {i}; \Theta \mid X _ {i}\right) \tag {9}
$$

by the Chain Rule for mutual information 7.

-  $I(\Theta; X^N, Y^N)$  - which measures how much information we store about our training data in the parameters of our encoder. This should also be minimized to ensure we learn compressed global representation, prevailing overfitting.

These mutual informations are all intractable in general, since we cannot compute the necessary marginals in closed form, given that we do not have access to the true data generating distribution.

# 2.1 FUNCTIONALS

Despite their intractability, we can compute variational bounds on these mutual informations.

# 2.1.1 ENTROPY

$$
S \equiv \left\langle \log \frac {p \left(\theta \mid x ^ {N} , y ^ {N}\right)}{q (\theta)} \right\rangle_ {P} \geq I (\Theta ; X ^ {N}, Y ^ {N}) \tag {10}
$$

The relative entropy in our parameters or just entropy for short measures the relative information between the distribution we assign our parameters in World  $P$  after learning from the data  $(X^{N},Y^{N})$ , with respect to some data independent  $q(\theta)$  prior on the parameters. This is an upper bound on the mutual information between the data and our parameters and as such can measure our risk of overfitting our parameters.

# 2.1.2 RATE

$$
R _ {i} \equiv \left\langle \log \frac {p \left(z _ {i} \mid x _ {i} , \theta\right)}{q \left(z _ {i}\right)} \right\rangle_ {P} \geq I \left(Z _ {i}; X _ {i}, \Theta\right) \tag {11}
$$

The rate measures the complexity of our representation. It is the relative information of a sample specific representation  $z_{i} \sim p(z|x_{i},\theta)$  with respect to our variational marginal  $q(z)$ . It measures how many bits we actually encode about each sample, and can measure how our risk of overfitting our representation. We use  $R \equiv \sum_{i}R_{i}$ .

# 2.1.3 CLASSIFICATION ERROR

$$
C _ {i} \equiv - \left\langle \log q \left(y _ {i} \mid z _ {i}\right) \right\rangle_ {P} \geq H \left(Y _ {i}\right) - I \left(Y _ {i}; Z _ {i}\right) = H \left(Y _ {i} \mid Z _ {i}\right) \tag {12}
$$

The classification error measures the conditional entropy of  $Y$  left after conditioning on  $Z$ . It is a measure of how much information about  $Y$  is left unspecified in our representation. This functional measures our supervised learning performance. We use  $C \equiv \sum_{i} C_{i}$ .

# 2.1.4 DISTORTION

$$
D _ {i} \equiv - \left\langle \log q \left(x _ {i} \mid z _ {i}\right) \right\rangle_ {P} \geq H \left(X _ {i}\right) - I \left(X _ {i}; Z _ {i}\right) = H \left(X _ {i} \mid Z _ {i}\right) \tag {13}
$$

The distortion measures the conditional entropy of  $X$  left after conditioning on  $Z$ . It is a measure of how much information about  $X$  is left unspecified in our representation. This functional measures our unsupervised learning performance. We use  $D \equiv \sum_{i} D_{i}$ .

# 2.2 GEOMETRY

The distributions  $p(z|x,\theta), p(\theta |x^N, y^N), q(z), q(x|z), q(y|z)$  can be chosen arbitrarily. Once chosen, the functionals  $R, C, D, S$  take on well described values. The choice of the five distributional families specifies a single point in a four-dimensional space.

Importantly, the sum of these functionals is a variational upper bound (up to an additive constant) for the minimum possible relative information between worlds (Appendix D):

$$
S + R + C + D \geq \mathcal {J} + \sum_ {i} H \left(X _ {i}, Y _ {i} \mid \Phi\right) \tag {14}
$$

Besides just the upper bound, we can consider the full space of feasible points. Notice that  $S$  and  $R$  are both themselves upper bounds on mutual informations, and so must be positive semi-definite. If our data is discrete, or if we have discretized it  $^8$ ,  $D$  and  $C$  which are both upper bounds on conditional entropies, must be positive as well. Along with Equation (14), given that  $\sum_{i} H(X_{i}, Y_{i}|\Phi)$  is a positive constant outside our control, the space of possible  $(R, C, D, S)$  values is at least restricted to be points in the positive orthant with some minimum possible Manhattan distance to the origin:

$$
S + R + C + D \geq \sum_ {i} H \left(X _ {i}, Y _ {i} \mid \Phi\right) \quad R \geq 0 \quad S \geq 0 \quad D \geq 0 \quad C \geq 0 \tag {15}
$$

Even in the infinite model family limit, data-processing inequalities on mutual information terms all defined in a set of variables that satisfy some nontrivial conditional dependencies ensure that there are regions in this functional space that are wholly out of reach. The surface of the feasible region maps an optimal frontier, optimal in the degree to which it minimizes mismatch between our two worlds subject to constraints on the relative magnitudes of the individual terms. This convex polytope has edges, faces and corners that are identifiable as the optimal solutions for well known objectives.

This story is a generalization of the story presented in Alemi et al. (2018), which can be considered a two-dimensional projection of this larger space (onto  $R, D$ ). Within our larger framework we can derive more specific bounds between subsets of the functionals. For instance:

$$
R _ {i} + D _ {i} \geq H \left(X _ {i}\right) + I \left(Z _ {i}; \Theta \mid X _ {i}\right). \tag {16}
$$

This mirrors the bound given in Alemi et al. (2018) where  $R + D \geq H(X)$ , which is still true given that all conditional mutual informations are positive semi-definite ( $H(X) + I(Z; \Theta|X) \geq H(X)$ ), but here we obtain a tighter pointwise bound that has a term measuring how much information about our encoding is revealed by the parameters after conditioning on the input itself. This term

$I(Z_{i};\Theta |X_{i})$  captures the degree to which our local representation is overly sensitive to the particular parameter settings  $^{910}$ .

# 2.3 GENERALIZATION

We can evaluate how much information our representations capture about the true data generating process. For instance,  $I(Z_{i};\Phi)$  which measures how much information about the true data generating procedure our local representations capture. Notice that given the conditional dependencies in world  $P$ , we have the following Markov chain:

$$
\Phi \rightarrow \left(X _ {i}, Y _ {i}, \Theta\right)\rightarrow Z _ {i} \tag {17}
$$

and so by the Data Processing Inequality (Cover & Thomas, 2012):

$$
I \left(Z _ {i}; \Phi\right) \leq I \left(Z _ {i}; \Theta , X _ {i}, Y _ {i}\right) = I \left(Z _ {i}; X _ {i}, \Theta\right) + \underline {{I \left(Z _ {i} ; Y _ {i} \mid X _ {i} , \Theta\right)}} \leq R _ {i}. \tag {18}
$$

The per-instance rate  $R_{i}$  forms an upper bound on the mutual information between our encoding  $Z_{i}$  and the true governing parameters of our data  $\Phi$ . Similarly, we can establish that:

$$
\Phi \rightarrow \left(X ^ {N}, Y ^ {N}\right)\rightarrow \Theta \Longrightarrow I (\Theta ; \Phi) \leq I (\Theta ; X ^ {N}, Y ^ {N}) \leq S. \tag {19}
$$

$S$  upper bounds the amount of information our encoder's parameters  $\Theta$ , the global representation of the dataset can contain about the true process  $\Phi$ . At the same time:

$$
I (\Theta ; \Phi) \leq I \left(X ^ {N}, Y ^ {N}; \Phi\right) \leq \sum_ {i} I \left(X _ {i}, Y _ {i}; \Phi\right), \tag {20}
$$

which sets a natural upper limit for the maximum  $S$  that might be useful.

# 3 OPTIMAL FRONTIER

As in Alemi et al. (2018), under mild assumptions about the variational distributional families, it can be argued that the surface is monotonic in all of its arguments. The optimal surface in the infinite family limit can be characterized as a convex polytope (Equation (15)). In practice we will be in the realistic setting corresponding to finite parametric families such as neural network approximators. We then expect that there is an irrevocable gap that opens up in the variational bounds. Any failure of the distributional families to model the correct corresponding marginal in  $P$  means that the space of all realizable  $R, C, D, S$  values will be some convex relaxation of the optimal feasible surface. This surface will be described some function  $f(R, C, D, S) = 0$ , which means we can identify points on the surface as a function of one functional with respect to the others (e.g.  $R = R(C, D, S)$ ). Finding points on this surface equates to solving a constrained optimization problem, e.g.

$$
\min  _ {q (z) q (x \mid z) q (y \mid z) p (z \mid x, \theta) p (\theta \mid \{x, y \})} R \text {s u c h t h a t} D = D _ {0}, S = S _ {0}, C = C _ {0}. \tag {21}
$$

Equivalently, we could solve the unconstrained Lagrange multipliers problem:

$$
\min  _ {q (z) q (x \mid z) q (y \mid z) p (z \mid x, \theta) p (\theta \mid \{x, y \})} R + \delta D + \gamma C + \sigma S. \tag {22}
$$

Here  $\delta, \gamma, \sigma$  are Lagrange multipliers that impose the constraints. They each correspond to the partial derivative of the rate at the solution with respect to their corresponding functional, keeping the others fixed.

Notice that this single objective encompasses a wide range of existing techniques.

- If we retain  $C$  alone, we are doing traditional supervised learning and our network will learn to be deterministic in its activations and parameters.

- If  $\delta = 0$  we no longer require a variational reconstruction network  $q(x|z)$ , and are doing some form of supervised learning generally.  
- If  $\delta = 0, \sigma = 0$  we exactly recover the Variational Information Bottleneck (VIB) objective of Alemi et al. (2016) (where  $\beta = 1 / \gamma$ ), a form of stochastically regularized supervised learning that imposes a bottleneck on how much information our representation can retain about the input, while simultaneously maximizing the amount of information the representation contains about the target.  
- If  $\delta = 0$  and  $\sigma, \gamma \to \infty$  but in such a way as to keep the ratio fixed  $\beta \equiv \sigma / \gamma$  (that is if we drop the  $R$  term and only keep  $C + \beta S$  as our objective) we recover the Information Bottleneck Lagrangian loss of Achille & Soatto (2017), presented as an alternative way to do Information Bottleneck (Tishby et al., 1999) but being stochastic on the parameters rather than the activations as in VIB.  
- As a special case, if our objective is set to  $C + S$  ( $\delta = 0, \sigma, \gamma \to \infty, \sigma / \gamma \to 1$ ), we obtain the objective for a Bayesian neural network, ala Blundell et al. (2015).  
- If we retain only  $D$ , we are training a stochastic autoencoder.  
- If  $\sigma = 0, \gamma = 0, \delta = 1$  the objective is equivalent to the ELBO used to train a VAE (Kingma & Welling, 2014).  
- If  $\sigma = 0, \gamma = 0$  more generally, the objective is equivalent to a  $\beta$ -VAE (Higgins et al., 2017) where  $\beta = 1/\delta$ .  
- If  $\gamma = 0$  all terms involving the auxiliary data  $Y$  drop out and we are doing some form of unsupervised learning without any variational classifier  $q(y|z)$ . The presence of the  $S$  term makes this more general than a usual  $\beta$ -VAE and should offer better generalization properties and control of overfitting by bottle-necking how much information we allow the parameters of our encoder to extract from the training data.  
-  $\sigma = 0, \gamma = \alpha, \delta = 1$  recovers the semi-supervised objective of Kingma et al. (2014).  
- In its most general form, in common parlance the full objective might be described as a temperature-regulated Bayesian semi-supervised  $\beta$ -VAE, or a Variational Information Bottleneck Lagrangian Autoencoder (VIBLA).

Examples of all of these objectives behavior on a simple toy model is shown in Appendix H.

Notice that all of these previous approaches describe low dimensional sub-surfaces of the optimal three-dimensional frontier. These approaches were all interested in different domains, some were focused on supervised prediction accuracy, others on learning a generative model. Depending on your specific problem, and downstream tasks, different points on the optimal frontier will be desirable. However, instead of choosing a single point on the frontier, we can now explore a region on the surface to see what class of solutions are possible within the modeling choices. By simply adjusting the three control parameters  $\delta, \gamma, \sigma$ , we can smoothly move across the entire frontier and smoothly interpolate between all of these objectives and beyond.

# 3.1 OPTIMIZATION

So far we've considered explicit forms of the objective in terms of the four functionals. For  $S$  this would require some kind of tractable approximation to the posterior over the parameters of our encoding distribution<sup>11</sup>. Alternatively, we can formally describe the exact solution to our minimization problem:

$$
\min  S \text {s . t .} R = R _ {0}, C = C _ {0}, D = D _ {0}. \tag {23}
$$

Recall that  $S$  measures the relative entropy of our parameter distribution with respect to the  $q(\theta)$  prior. As such, the solution that minimizes the relative entropy subject to some constraints is a generalized Boltzmann distribution (Jaynes, 1957):

$$
p ^ {*} (\theta | \{x, y \}) = \frac {q (\theta)}{\mathcal {Z}} e ^ {- (R + \delta D + \gamma C) / \sigma}. \tag {24}
$$

Here  $\mathcal{Z}$  is the partition function, the normalization constant for the distribution

$$
\mathcal {Z} = \int d \theta q (\theta) e ^ {- (R + \delta D + \gamma C) / \sigma} \tag {25}
$$

This suggests an alternative method for finding points on the optimal frontier. We could turn the unconstrained Lagrange optimization problem that required some explicit choice of tractable posterior distribution over parameters into a sampling problem for a richer implicit distribution.

A naive way to draw samples from this posterior would be to use Stochastic Gradient Langevin Dynamics or its cousins (Welling & Teh, 2011; Chen et al., 2014; Ma et al., 2015) which, in practice, would look like ordinary stochastic gradient descent (or its cousins like momentum) for the objective  $R + \delta D + \gamma C$ , with injected noise. By choosing the magnitude of the noise relative to the learning rate, the effective temperature  $\sigma$  can be controlled.

There is increasing evidence that the stochastic part of stochastic gradient descent itself is enough to turn SGD less into an optimization procedure and more into an approximate posterior sampler (Mandt et al., 2017; Smith & Le, 2017; Achille & Soatto, 2017; Zhang et al., 2018; Chaudhari & Soatto, 2017), where hyperparameters such as the learning rate and batch size set the effective temperature. If ordinary stochastic gradient descent is doing something more akin to sampling from a posterior and less like optimizing to some minimum, it would help explain improved performance through ensemble averages of different points along trajectories (Huang et al., 2017).

When viewed in this light, Equation 24 describes the optimal posterior for the parameters so as to ensure the minimal divergence between worlds  $P$  and  $Q$ .  $q(\theta)$  plays the role of the prior over parameters, but our overall objective is minimized when

$$
q (\theta) = p (\theta) = \left\langle p \left(\theta \mid x ^ {N}, y ^ {N}\right) \right\rangle_ {p \left(x ^ {N}, y ^ {N}\right)}. \tag {26}
$$

That is, when our prior is the marginal of the posteriors over all possible datasets drawn from the true distribution. A fair draw from this marginal is to take a sample from the posterior obtained on a different but related dataset. Insomuch as ordinary SGD training is an approximate method for drawing a posterior sample, the common practice of fine-tuning a pretrained network on a related dataset is using a sample from the optimal prior as our initial parameters. The fact that fine-tuning approximates use of an optimal prior presumably helps explain its broad success.

If we identify our true goal not as optimizing some objective but instead directly sampling from Equation 24, we can consider alternative approaches to define our learning dynamics, such as parallel tempering or population annealing (Machta & Ellis, 2011). Alternatively, we could, instead of adopting variational bounds on the mutual informations, consider other mutual information bounds such as those in Ishmael Belghazi et al. (2018); van den Oord et al. (2018). Perhaps our priors can be fit, providing we form estimates of the expectation over datasets (e.g. bootstrapping or jackknifing our dataset (DasGupta, 2008)).

# 4 THERMODYNAMICS

So far we have described a framework for learning that involves finding points that lie on the surface of a convex three-dimensional surface in terms of four functional coordinates  $R, C, D, S$ . Interestingly, this is all that is required to establish a formal connection to thermodynamics, which similarly is little more than the study of exact differentials (Sethna, 2006; Finn, 1993).

Whereas previous approaches connecting thermodynamics and learning (Parrondo et al., 2015; Still, 2017; Still et al., 2012) have focused on describing the thermodynamics and statistical mechanics of physical realizations of learning systems (i.e. the heat bath in these papers is a physical heat bath at finite temperature), in this work we make a formal analogy to the structure of the theory of thermodynamics, without any physical content.

# 4.1 FIRST LAW OF LEARNING

The optimal frontier creates an equivalence class of states, being the set of all states that minimize as much as possible the distortion introduced in projecting world  $P$  onto a set of distributions that

respect the conditions in  $Q$ . The surface satisfies some equation  $f(R, C, D, S) = 0$  which we can use to describe any one of these functionals in terms of the rest, e.g.  $R = R(C, D, S)$ . This function is entire, and so we can equate partial derivatives of the function with differentials of the functionals<sup>12</sup>:

$$
d R = \left(\frac {\partial R}{\partial C}\right) _ {D, S} d C + \left(\frac {\partial R}{\partial D}\right) _ {C, S} d D + \left(\frac {\partial R}{\partial S}\right) _ {C, D} d S. \tag {27}
$$

Since the function is smooth and convex, instead of identifying the surface of optimal rates in terms of the functionals  $C, D, S$ , we could just as well describe the surface in terms of the partial derivatives by applying a Legendre transformation. We will name the partial derivatives:

$$
\gamma \equiv - \left(\frac {\partial R}{\partial C}\right) _ {D, S} \quad \delta \equiv - \left(\frac {\partial R}{\partial D}\right) _ {C, S} \quad \sigma \equiv - \left(\frac {\partial R}{\partial S}\right) _ {C, D}. \tag {28}
$$

These measure the exchange rate for turning rate into reduced distortion, reduced classification error, or increased entropy, respectively.

The functionals  $R, C, D, S$  are analogous to extensive thermodynamic variables such as volume, entropy, particle number, magnetic field, charge, surface area, length and energy which grow as the system grows, while the named partial derivatives  $\gamma, \delta, \sigma$  are analogous to the intensive, generalized forces in thermodynamics corresponding to their paired state variable, such as pressure, temperature, chemical potential, magnetization, electromotive force, surface tension, elastic force, etc. Just as in thermodynamics, the extensive functionals are defined for any state, while the intensive partial derivatives are only well defined for equilibrium states, which in our language are the states lying on the optimal surface<sup>13</sup>.

Recasting our total differential:

$$
d R = - \gamma d C - \delta d D - \sigma d S, \tag {29}
$$

we create a law analogous to the First Law of Thermodynamics. In thermodynamics the First Law is often taken to be a statement about the conservation of energy, and by analogy here we could think about this law as a statement about the conservation of information. Granted, the actual content of the law is fairly vacuous, equivalent only to the statement that there exists a scalar function  $R = R(C, D, S)$  defining our surface and its partial derivatives.

# 4.2 MAXWELL RELATIONS AND THERMODYNAMIC POTENTIALS

Requiring that Equation 29 be an exact differential has mathematically trivial but intuitively non-obvious implications that relate various partial derivatives of the system to one another, akin to the Maxwell Relations in thermodynamics. For example, requiring that mixed second partial derivatives are symmetric establishes that:

$$
\left(\frac {\partial^ {2} R}{\partial D \partial C}\right) = \left(\frac {\partial^ {2} R}{\partial C \partial D}\right) \Rightarrow \left(\frac {\partial \delta}{\partial C}\right) _ {D} = \left(\frac {\partial \gamma}{\partial D}\right) _ {C}. \tag {30}
$$

This equates the result of two very different experiments. In the experiment encoded in the partial derivative on the left, one would measure the change in the derivative of the  $R - D$  curve  $(\delta)$  as a function of the classification error  $(C)$  at fixed distortion  $(D)$ . On the right one would measure the change in the derivative of the  $R - C$  curve  $(\gamma)$  as a function of the distortion  $(D)$  at fixed classification error  $(C)$ . As different as these scenarios appear, they are mathematically equivalent. A full set of Maxwell relations can be found in Appendix F.

We can additionally take and name higher order partial derivatives, analogous to the susceptibilities of thermodynamics like bulk modulus, the thermal expansion coefficient, or heat capacities. For instance, we can define the analog of heat capacity for our system, a sort of rate capacity at constant distortion:

$$
K _ {D} \equiv \left(\frac {\partial R}{\partial \sigma}\right) _ {D}. \tag {31}
$$

Just as in thermodynamics, these susceptibilities may offer useful ways to characterize and quantify the systematic differences between model families. Perhaps general scaling laws can be found between susceptibilities and network widths, or depths, or number of parameters or dataset size. Divergences or discontinuities in the susceptibilities are the hallmark of phase transitions in physical systems, and it is reasonable to expect to see similar phenomenon for certain models.

A great deal of first, second and third order partial derivatives in thermodynamics are given unique names. This is because the quantities are particularly useful for comparing different physical systems. We expect a subset of the first, second and higher order partial derivatives of the base functionals will prove similarly useful for comparing, quantifying, and understanding differences between modeling choices.

# 4.3 SECOND LAW OF LEARNING?

Even when doing deterministic training, training is non-invertible (Maclaurin et al., 2015), and we need to contend with and track the entropy  $(S)$  term. We set the parameters of our networks initially with a fair draw from some prior distribution  $q(\theta)$ . The training procedure acts as a Markov process on the distribution of parameters, transforming it from the prior distribution into some modified distribution, the posterior  $p(\theta | x^N, y^N)$ . Optimization is a many-to-one function, that in the ideal limiting case, maps all possible initializations to a single global optimum. In this limiting case  $S$  would be divergent, and there is nothing to prevent us from memorizing the training set.

The Second Law of Thermodynamics states that the entropy of an isolated system tends to increase. All systems tend to disorder, and this places limits on the maximum possible efficiency of heat engines.

Formally, there are many statements akin to the Second Law of Thermodynamics that can be made about Markov chains generally (Cover & Thomas, 2012). The central one is that for any for any two distributions  $p_n, q_n$  both evolving according to the same Markov process ( $n$  marks the time step), the relative entropy  $D_{\mathrm{KL}}[p_n; q_n]$  is monotonically decreasing with time. This establishes that for a stationary Markov chain, the relative entropy to the stationary state  $D_{\mathrm{KL}}[p_n; p_\infty]$  monotonically decreases<sup>14</sup>.

In our language, we can make strong statements about dynamics that target points on the optimal frontier, or dynamics that implement a relaxation towards equilibrium. There is a fundamental distinction between states that live on the frontier and those off of it, analogous to the distinction between equilibrium and non-equilibrium states in thermodynamics.

Any equilibrium distribution can be expressed in the form Equation (24) and identified by its partial derivatives  $\gamma, \delta, \sigma$ . If name the objective in Equation (22):

$$
J (\gamma , \delta , \sigma) \equiv R + \delta D + \gamma C + \sigma S, \tag {32}
$$

The value this objective takes for any equilibrium distribution can be shown to be given by the log partition function (Equation (25)):

$$
\min  J (\gamma , \delta , \sigma) = - \sigma \log \mathcal {Z} (\gamma , \delta , \sigma) \tag {33}
$$

and the KL divergence between any distribution over parameters  $p(\theta)$  and an equilibrium distribution is:

$$
D _ {\mathrm {K L}} [ p (\theta); p ^ {*} (\theta ; \gamma , \delta , \sigma) ] = \Delta J / \sigma \tag {34}
$$

$$
\Delta J \equiv J ^ {\text {n o n e q}} (p; \gamma , \delta , \sigma) - J (\gamma , \delta , \sigma) \tag {35}
$$

Where  $J^{\mathrm{noneq}}$  is the non-equilibrium objective:

$$
J ^ {\text {n o n e q}} (p; \gamma , \delta , \sigma) = \langle R + \delta D + \gamma C + \sigma S \rangle_ {p (\theta)}. \tag {36}
$$

For a stationary Markov process whose stationary distribution is an equilibrium distribution the KL divergence to the stationary distribution must monotonically decrease each step. This means the  $\Delta J / \sigma$  must decrease monotonically, that is our objective  $J$  must decrease monotonically:

$$
J _ {t = 0} \geq J _ {t} \geq J _ {t + 1} \geq J _ {t = \infty}. \tag {37}
$$

Furthermore, if we use  $q(\theta)$  as our prior over parameters, we know:

$$
J _ {t = 0} = \left\langle R + \delta D + \gamma C \right\rangle_ {q (\theta)} \tag {38}
$$

$$
J _ {t = \infty} = - \sigma \log Z. \tag {39}
$$

# 5 CONCLUSION

We have formalized representation learning as the process of minimizing the distortion introduced when we project the real world (World  $P$ ) onto the world we desire (World  $Q$ ). The projection is naturally described by a set of four functionals which variationally bound relevant mutual informations in the real world. Relations between the functionals describe an optimal three-dimensional surface in a four dimensional space of optimal states. A single learning objective targeting points on this optimal surface can express a wide array of existing learning objectives spanning from unsupervised learning to supervised learning and everywhere in between. The geometry of the optimal frontier suggests a wide array of identities involving the functionals and their partial derivatives. This offers a direct analogy to thermodynamics independent of any physical content. By analogy to thermodynamics, we can begin to develop new quantitative measures and relationships amongst properties of our models that we believe will offer a new class of theoretical understanding of learning behavior.

# REFERENCES

L Accardi. De Finetti, 2018. URL http://www.encyclopediaofmath.org/index.php?title=De_Finetti_theorem&oldid=12884.  
A. Achille and S. Soatto. Emergence of Invariance and Disentangling in Deep Representations. Proceedings of the ICML Workshop on Principled Approaches to Deep Learning, 2017.  
Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. arXiv:1612.00410, 2016. URL http://arxiv.org/abs/1612.00410.  
Alexander A Alemi, Ben Poole, Joshua V Dillon, Rif A Saurous, and Kevin Murphy. Fixing a broken ELBO. ICML 2018, 2018. URL http://arxiv.org/abs/1711.00464.  
William Bialek, Ilya Nemenman, and Naftali Tishby. Predictability, complexity, and learning. Neural computation, 13(11):2409-2463, 2001.  
C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra. Weight Uncertainty in Neural Networks. arXiv: 1505.05424, May 2015. URL https://arxiv.org/abs/1505.05424.  
Pratik Chaudhari and Stefano Soatto. Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks. arXiv, 2017. URL https://arxiv.org/abs/1710.11029.  
T. Chen, E. B. Fox, and C. Guestrin. Stochastic Gradient Hamiltonian Monte Carlo. arXiv:1402.4102, February 2014. URL https://arxiv.org/abs/1402.4102.  
X. Chen, D. P. Kingma, T. Salimans, Y. Duan, P. Dhariwal, J. Schulman, I. Sutskever, and P. Abbeel. Variational Lossy Autoencoder. arXiv, 2016. URL https://arxiv.org/abs/1611.02731.  
Thomas M Cover and Joy A Thomas. Elements of information theory. John Wiley & Sons, 2012.  
Imre Csiszár and František Matúš. Information projections revisited. IEEE Transactions on Information Theory, 49(6):1474-1490, 2003.  
Anirban DasGupta. Edgeworth expansions and cumulants. In Asymptotic Theory of Statistics and Probability, pp. 185-201. Springer, 2008.  
Colin BP Finn. Thermal physics. CRC Press, 1993.  
Nir Friedman, Ori Mosenzon, Noam Slonim, and Naftali Tishby. Multivariate information bottleneck. In Proceedings of the Seventeenth conference on Uncertainty in artificial intelligence, pp. 152-161. Morgan Kaufmann Publishers Inc., 2001.

Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner.  $\beta$ -VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. 2017.  
G. Huang, Y. Li, G. Pleiss, Z. Liu, J. E. Hopcroft, and K. Q. Weinberger. Snapshot Ensembles: Train 1, get M for free. arXiv: 1704.00109, March 2017. URL https://arxiv.org/abs.1704.00109.  
M. Ishmael Belghazi, A. Baratin, S. Rajeswar, S. Ozair, Y. Bengio, A. Courville, and R Devon Hjelm. MINE: Mutual Information Neural Estimation. arXiv, 2018. URL https://arxiv.org/abs/1801.04062.  
Edwin T Jaynes. Information theory and statistical mechanics. Physical review, 106(4):620, 1957.  
D. P. Kingma, D. J. Rezende, S. Mohamed, and M. Welling. Semi-Supervised Learning with Deep Generative Models. arXiv: 1406.5298, June 2014. URL https://arxiv.org/abs/1406.5298.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. 2014.  
Y.-A. Ma, T. Chen, and E. B. Fox. A Complete Recipe for Stochastic Gradient MCMC. arXiv:1506.04696, June 2015. URL https://arxiv.org/abs/1506.04696.  
J. Machta and R. S. Ellis. Monte Carlo Methods for Rough Free Energy Landscapes: Population Annealing and Parallel Tempering. Journal of Statistical Physics, 144:541-553, August 2011. doi: 10.1007/s10955-011-0249-0. URL https://arxiv.org/abs/1104.1138.  
Dougal Maclaurin, David Duvenaud, and Ryan P. Adams. Gradient-based hyperparameter optimization through reversible learning. In Proceedings of the 32Nd International Conference on International Conference on Machine Learning - Volume 37, ICML'15, pp. 2113-2122. JMLR.org, 2015. URL http://dl.acm.org/citation.cfm?id=3045118.3045343.  
S. Mandt, M. D. Hoffman, and D. M. Blei. Stochastic Gradient Descent as Approximate Bayesian Inference. arXiv: 1704.04289, April 2017. URL https://arxiv.org/abs/1704.04289.  
Juan MR Parrondo, Jordan M Horowitz, and Takahiro Sagawa. Thermodynamics of information. Nature physics, 11(2):131-139, 2015. URL http://jordanmhorowitz.mit.edu/sites/default/files/documents/natureInfo.pdf.  
James Sethna. Statistical mechanics: entropy, order parameters, and complexity, volume 14. Oxford University Press, 2006. URL http://pages.physics.cornell.edu/~sethna/ StatMech/EntropyOrderParametersComplexity.pdf.  
Noam Slonim, Gurinder S Atwal, Gasper Tkacik, and William Bialek. Estimating mutual information and multi-information in large networks. arXiv, 2005. URL https://arxiv.org/abs/cs/0502017.  
S. L. Smith and Q. V. Le. A Bayesian Perspective on Generalization and Stochastic Gradient Descent. arXiv:1710.06451, October 2017. URL https://arxiv.org/abs/1710.06451.  
S. Still. Thermodynamic cost and benefit of data representations. arXiv: 1705.00612, April 2017. URL https://arxiv.org/abs/1705.00612.  
S. Still, D. A. Sivak, A. J. Bell, and G. E. Crooks. Thermodynamics of Prediction. Physical Review Letters, 109(12):120604, September 2012. doi: 10.1103/PhysRevLett.109.120604. URL https://arxiv.org/abs/1203.3271.  
Susanne Still. Information bottleneck approach to predictive inference. Entropy, 16(2):968-989, 2014.  
N. Tishby, F.C. Pereira, and W. Biale. The information bottleneck method. In *The 37th annual Allerton Conf. on Communication, Control, and Computing*, pp. 368-377, 1999. URL https://arxiv.org/abs/physics/0004057.

A. van den Oord, Y. Li, and O. Vinyals. Representation Learning with Contrastive Predictive Coding. arXiv, 2018. URL https://arxiv.org/abs/1807.03748.  
Sumio Watanabe. *Algebraic geometry and statistical learning theory*, volume 25. Cambridge University Press, 2009.  
Sumio Watanabe. Mathematical theory of Bayesian statistics. CRC Press, 2018.  
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681-688, 2011.  
Y. Zhang, A. M. Saxe, M. S. Advani, and A. A. Lee. Energy-entropy competition and the effectiveness of stochastic gradient descent in machine learning. arXiv: 1803.01927, March 2018. URL https://arxiv.org/abs/1803.01927.
