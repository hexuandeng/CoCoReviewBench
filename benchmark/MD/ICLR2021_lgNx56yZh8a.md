# BAYESIAN FEW-SHOT CLASSIFICATION WITH ONE-VS-EACH POLYA-GAMMA AUGMENTED GAUSSIAN PROCESSES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Few-shot classification (FSC), the task of adapting a classifier to unseen classes given a small labeled dataset, is an important step on the path toward human-like machine learning. Bayesian methods are well-suited to tackling the fundamental issue of overfitting in the few-shot scenario because they allow practitioners to specify prior beliefs and update those beliefs in light of observed data. Contemporary approaches to Bayesian few-shot classification maintain a posterior distribution over model parameters, which is slow and requires storage that scales with model size. Instead, we propose a Gaussian process classifier based on a novel combination of Pólya-gamma augmentation and the one-vs-each softmax approximation (Titsias, 2016) that allows us to efficiently marginalize over functions rather than model parameters. We demonstrate improved accuracy and uncertainty quantification on both standard few-shot classification benchmarks and few-shot domain transfer tasks.

# 1 INTRODUCTION

Few-shot classification (FSC) is a rapidly growing area of machine learning that seeks to build classifiers able to adapt to novel classes given only a few labeled examples. It is an important step towards machine learning systems that can successfully handle challenging situations such as personalization, rare classes, and time-varying distribution shift. The shortage of labeled data in FSC leads to uncertainty over the parameters of the model, known as model uncertainty or epistemic uncertainty. If model uncertainty is not handled properly in the few-shot setting, there is a significant risk of overfitting. In addition, FSC is increasingly being used for risk-averse applications such as medical diagnosis (Prabhu, 2019) and human-computer interfaces (Wang et al., 2019) where it is important for a few-shot classifier to know when it is uncertain.

Bayesian methods maintain a distribution over model parameters and thus provide a natural framework for capturing this inherent model uncertainty. In a Bayesian approach, a prior distribution is first placed over the parameters of a model. After data is observed, the posterior distribution over parameters is computed using Bayesian inference. This elegant treatment of model uncertainty has led to a surge of interest in Bayesian approaches to FSC that infer a posterior distribution over the weights of a neural network (Finn et al., 2018; Yoon et al., 2018; Ravi & Beatson, 2019).

Although conceptually appealing, there are several practical obstacles to applying Bayesian inference directly to the weights of a neural network. Bayesian neural networks (BNNs) are expensive from both a computational and memory perspective. Moreover, specifying meaningful priors in parameter space is known to be difficult due to the complex relationship between weights and network outputs (Sun et al., 2019). Gaussian processes (GPs) instead maintain a distribution over functions rather than model parameters. The prior is directly specified by a mean and covariance function, which may be parameterized by deep neural networks. When used with Gaussian likelihoods, GPs admit closed form expressions for the posterior and predictive distributions. They exchange the computational drawbacks of BNNs for cubic scaling with the number of examples. In FSC, where the number of examples is small, this is often an acceptable trade-off.

When applying GPs to classification with a softmax likelihood, the non-conjugacy of the GP prior renders posterior inference intractable. Many approximate inference methods have been proposed

to circumvent this, including variational inference and expectation propagation. In this paper we investigate a particularly promising class of approaches that augments the GP model with a set of auxiliary random variables, such that when the auxiliary variables are marginalized out the original model is recovered (Albert & Chib, 1993; Girolami & Rogers, 2006; Linderman et al., 2015). Such augmentation-based approaches typically admit efficient Gibbs-sampling procedures for generating posterior samples which when combined with Fisher's identity (Douc et al., 2014) can be used to optimize the parameters of the mean and covariance functions.

In particular, augmentation with Pólya-gamma random variables (Polson et al., 2013) makes inference tractable in logistic models. Naively, this is useful for handling binary classification, but in this paper we show how to extend this augmentation to classification with multiple classes by using the one-vs-each softmax approximation (Titsias, 2016), which can be expressed as a product of logistic sigmoidals.

In this work, we make several contributions:

- We propose a novel GP classification method that combines the one-vs-each softmax approximation (Titsias, 2016) with Pólya-gamma augmentation for tractable inference.  
- We demonstrate competitive classification accuracy of our method on standard FSC benchmarks and challenging domain transfer settings.  
- We propose several new benchmarks for uncertainty quantification in FSC, including calibration, robustness to input noise, and out-of-episode detection.  
- We demonstrate improved uncertainty quantification of our method on the proposed benchmarks relative to standard few-shot baselines.

# 2 RELATED WORK

Our work is related to both GP classification methods for handling non-conjugacy of classification likelihoods and Bayesian approaches to FSC. We summarize the most relevant work here.

# 2.1 GP CLASSIFICATION

Non-augmentation approaches. There are several classes of approaches for applying Gaussian processes to classification. The most straightforward method, known as least squares classification (Rifkin & Klautau, 2004; Williams & Rasmussen, 2006) treats class labels as real-valued observations with a Gaussian likelihood. Laplace's approximation constructs a Gaussian approximate posterior centered at the posterior mode. Variational approaches (Titsias, 2009) maximize a lower bound on the log marginal likelihood. In expectation propagation (Minka, 2001; Kim & Ghahramani, 2006), local Gaussian approximations to the likelihood are fitted iteratively until convergence.

Augmentation approaches. Augmentation-based approaches introduce auxiliary random variables such that the original model is recovered when marginalized out. Girolami & Rogers (2006) propose a Gaussian augmentation for multinomial probit regression. Linderman et al. (2015) utilize Pólya-gamma augmentation (Polson et al., 2013) and a stick-breaking construction to decompose a multinomial distribution into a product of binomials. Galy-Fajou et al. (2019) proposes a logistic-softmax likelihood for classification and uses Gamma and Poisson augmentation in addition to Pólya-gamma augmentation in order to perform inference.

# 2.2 FEW-SHOT CLASSIFICATION

Meta-learning. A common approach to FSC is meta-learning, which seeks to learn how to update neural network parameters. The Meta-learner LSTM (Ravi & Larochelle, 2017) learns a meta-level LSTM to recurrently output a new set of parameters for the base learner. MAML (Finn et al., 2017) learns deep neural networks to perform well on the task-specific loss after one or a few steps of gradient descent on the support set by directly backpropagating through the gradient descent procedure itself. LEO (Rusu et al., 2018) performs meta-learning in a learned low-dimensional latent space from which the parameters of a classifier are generated.

Metric learning. Metric learning approaches learn distances such that input examples can be meaningfully compared. Siamese Networks (Koch et al., 2015) learn a shared embedding network along with a distance layer for computing the probability that two examples belong to the same class. Matching Networks (Vinyals et al., 2016) uses a nonparametric classification in the form of attention over nearby examples, which can be interpreted as a form of soft  $k$ -nearest neighbors in the embedding space. Prototypical Networks (Snell et al., 2017) make predictions based on distances to nearest class centroids. Relation Networks (Sung et al., 2018) instead learn a more complex neural network distance function on top of the embedding layer.

Bayesian Few-shot Classification. More recently, Bayesian FSC approaches that attempt to infer a posterior over task-specific parameters have appeared. Grant et al. (2018) reinterprets MAML as an approximate empirical Bayes algorithm and propose LLAMA, which optimizes the Laplace approximation to the marginal likelihood. Bayesian MAML (Yoon et al., 2018) instead uses Stein Variational Gradient Descent (SVGD) (Liu & Wang, 2016) to approximate the posterior distribution over model parameters. VERSA (Gordon et al., 2019) uses amortized inference networks to obtain an approximate posterior distribution over task-specific parameters. ABML (Ravi & Beatson, 2019) uses a few steps of Bayes by Backprop (Blundell et al., 2015) on the support set to produce an approximate posterior over network parameters.

GPs for Few-shot Learning. There have been relatively few works applying GPs to few-shot learning. Tossou et al. (2019) consider Gaussian processes in the context of few-shot regression with Gaussian likelihoods. GPNet (Patacchiola et al., 2019) use Gaussian processes with least squares classification to perform few-shot classification and learn covariance functions parameterized by deep neural networks. More recently, Titsias et al. (2020) applies GPs to meta-learning with the variational information bottleneck.

# 3 BACKGROUND

In this section we review Pólya-gamma augmentation for binary classification and the one-vs-each approximation before introducing our method in Section 4.

# 3.1 PÓLYA-GAMMA AUGMENTATION

Suppose we have a vector of logits  $\psi \in \mathbb{R}^N$  with corresponding binary labels  $\mathbf{y} \in \{0,1\}^N$ . The logistic likelihood is

$$
p (\mathbf {y} \mid \psi) = \prod_ {i = 1} ^ {N} \sigma \left(\psi_ {i}\right) ^ {y _ {i}} \left(1 - \sigma \left(\psi_ {i}\right)\right) ^ {1 - y _ {i}} = \prod_ {i = 1} ^ {N} \frac {\left(e ^ {\psi_ {i}}\right) ^ {y _ {i}}}{1 + e ^ {\psi_ {i}}}, \tag {1}
$$

where  $\sigma(\cdot)$  is the logistic sigmoid function. Let the prior over  $\psi$  be Gaussian:  $p(\psi) = \mathcal{N}(\psi|\mu, \Sigma)$ . In Bayesian inference, we are interested in the posterior  $p(\psi|\mathbf{y}) \propto p(\mathbf{y}|\psi)p(\psi)$  but the form of (1) does not admit analytic computation due to non-conjugacy. The main idea of Pólya-gamma augmentation is to introduce auxiliary random variables  $\omega$  to the likelihood such that the original model is recovered when  $\omega$  is marginalized out:  $p(\mathbf{y}|\psi) = \int p(\omega)p(\mathbf{y}|\psi,\omega)d\omega$ . Conditioned on  $\omega \sim \mathrm{PG}(b,c)$ , the likelihood is proportional to a diagonal Gaussian (see Section A for a full derivation):

$$
p (\mathbf {y} | \boldsymbol {\psi}, \boldsymbol {\omega}) \propto \prod_ {i = 1} ^ {N} e ^ {- \omega_ {i} \psi_ {i} ^ {2} / 2} e ^ {\kappa_ {i} \psi_ {i}} \propto \mathcal {N} \left(\boldsymbol {\Omega} ^ {- 1} \boldsymbol {\kappa} | \boldsymbol {\psi}, \boldsymbol {\Omega} ^ {- 1}\right), \tag {2}
$$

where  $\kappa_{i} = y_{i} - 1 / 2$  and  $\Omega = \mathrm{diag}(\omega)$ . The conditional distribution over  $\psi$  given  $\mathbf{y}$  and  $\omega$  is now tractable:

$$
p (\boldsymbol {\psi} | \mathbf {y}, \boldsymbol {\omega}) \propto p (\mathbf {y} | \boldsymbol {\psi}, \boldsymbol {\omega}) p (\boldsymbol {\psi}) \propto \mathcal {N} (\boldsymbol {\psi} | \tilde {\boldsymbol {\Sigma}} (\boldsymbol {\Sigma} ^ {- 1} \boldsymbol {\mu} + \boldsymbol {\kappa}), \tilde {\boldsymbol {\Sigma}}), \tag {3}
$$

where  $\tilde{\Sigma} = (\Sigma^{-1} + \Omega)^{-1}$ . The conditional distribution of  $\omega$  given  $\psi$  and  $\mathbf{y}$  can also be easily computed:

$$
p \left(\omega_ {i} \mid y _ {i}, \psi_ {i}\right) \propto \operatorname {P G} \left(\omega_ {i} \mid 1, 0\right) e ^ {- \omega_ {i} \psi_ {i} ^ {2} / 2} \propto \operatorname {P G} \left(\omega_ {i} \mid 1, \psi_ {i}\right), \tag {4}
$$

where the last expression follows from the exponential tilting property of Pólya-gamma random variables. This suggests a Gibbs sampling procedure in which iterates  $\pmb{\omega}^{(t)}\sim p(\pmb {\omega}|\mathbf{y},\pmb{\psi}^{(t - 1)})$  and

$\psi^{(t)} \sim p(\psi | \mathbf{X}, \mathbf{y}, \boldsymbol{\omega}^{(t)})$  are drawn sequentially until the Markov chain reaches its stationary distribution, which is the joint posterior  $p(\psi, \boldsymbol{\omega} | \mathbf{y})$ . Fortunately, efficient samplers for the Polya-gamma distribution have been developed (Windle et al., 2014) to facilitate this.

# 3.2 ONE-VS-EACH APPROXIMATION TO SOFTMAX

The one-vs-each (OVE) approximation (Titsias, 2016) was formulated as a lower bound to the softmax likelihood in order to handle classification over a large number of output classes, where computation of the normalizing constant is prohibitive. We use the OVE approximation not to deal with extreme classification, but rather due to its compatibility with Pólya-gamma augmentation, as we shall soon see. The one-vs-each approximation can be derived by first rewriting the softmax likelihood as follows:

$$
p ^ {\mathrm {S M}} (y = c | \mathbf {f}) \triangleq \frac {e ^ {f _ {c}}}{\sum_ {c ^ {\prime}} e ^ {f _ {c ^ {\prime}}}} = \frac {1}{1 + \sum_ {c ^ {\prime} \neq c} e ^ {- \left(f _ {c} - f _ {c ^ {\prime}}\right)}}, \tag {5}
$$

where  $\mathbf{f} \triangleq (f_1, \ldots, f_C)^\top$  are the logits. Since  $\prod_i (1 + \alpha_i) \geq (1 + \sum_i \alpha_i)$  for  $\alpha_i \geq 0$ , the softmax likelihood (5) can be bounded as follows:

$$
p ^ {\mathrm {S M}} (y = c \mid \mathbf {f}) \geq \prod_ {c ^ {\prime} \neq c} \frac {1}{1 + e ^ {- \left(f _ {c} - f _ {c ^ {\prime}}\right)}} = \prod_ {c ^ {\prime} \neq c} \sigma \left(f _ {c} - f _ {c ^ {\prime}}\right), \tag {6}
$$

which is the OVE lower bound. This expression avoids the normalizing constant and factorizes into a product of pairwise sigmoids.

# 4 ONE-VS-EACH PÓLYA-GAMMA GPS

We now introduce our method for GP-based Bayesian few-shot classification, which utilizes a novel combination of Pólya-gamma augmentation and the one-vs-each (OVE) approximation.

# 4.1 OVE AS A LIKELIHOOD FUNCTION

Suppose we have access to examples  $\mathbf{X} \in \mathbb{R}^{N \times D}$  with corresponding one-hot labels  $\mathbf{Y} \in \{0,1\}^{N \times C}$ , where  $C$  is the number of classes. We consider the logits jointly as a single vector  $\mathbf{f} \triangleq (f_1^1, \ldots, f_N^1, f_1^2, \ldots, f_N^2, \ldots, f_1^C, \ldots, f_N^C)^\top$  and place an independent GP prior on the logits for each class:  $\mathbf{f}^c(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ . Therefore we have  $p(\mathbf{f}|\mathbf{X}) = \mathcal{N}(\mathbf{f}|\boldsymbol{\mu}, \mathbf{K})$ , where  $\mu_i^c = m(\mathbf{x}_i)$  and  $\mathbf{K}$  is block diagonal with  $K_{ij}^c = k(\mathbf{x}_i, \mathbf{x}_j)$  for each block  $\mathbf{K}^c$ .

The Pólya-gamma integral identity used to derive (2) does not have a multi-class analogue and thus a direct application of the augmentation scheme to the softmax likelihood is nontrivial. Instead, we propose to directly replace the softmax with an OVE-based likelihood function, which is the same as (6):

$$
p ^ {\mathrm {O V E}} \left(y _ {i} = c \mid \mathbf {f} _ {i}\right) \triangleq \prod_ {c ^ {\prime} \neq c} \sigma \left(f _ {i} ^ {c} - f _ {i} ^ {c ^ {\prime}}\right). \tag {7}
$$

We use this likelihood not to handle extreme classification as Titsias (2016), but instead due to its close relationship with the softmax likelihood while maintaining tractable inference with Pólya-gamma augmentation. Compared to other choices of likelihoods used by previous approaches, there are several reasons to prefer OVE. Compared to the Gaussian augmentation approach of Girolami & Rogers (2006), Pólya-gamma augmentation has the benefit of fast mixing and moreover a single value of  $\omega$  can capture much of the marginal distribution over function values<sup>1</sup>. The stick-breaking construction of Linderman et al. (2015) induces a dependence on the ordering of classes, which leads to undesirable asymmetry. Finally, the logistic-softmax likelihood of Galy-Fajou et al. (2019) requires three augmentations and careful learning of the mean function to avoid a priori underconfidence (see Section I for more details).

# 4.2 POSTERIOR INFERENCE VIA GIBBS SAMPLING

Define the matrix  $\mathbf{A} \triangleq \mathrm{OVE-Matrix}(\mathbf{Y})$  to be a  $CN \times CN$  sparse block matrix with  $C$  row partitions and  $C$  column partitions. Each block  $\mathbf{A}_{cc'}$  is a diagonal  $N \times N$  matrix defined as follows:

$$
\mathbf {A} _ {c c ^ {\prime}} \triangleq \operatorname {d i a g} \left(\mathbf {Y}. c\right) - \mathbb {1} [ c = c ^ {\prime} ] \mathbf {I} _ {n}, \tag {8}
$$

where  $\mathbf{Y}_{\cdot c}$  denotes the  $c$ th column of  $\mathbf{Y}$ . Now the binary logit vector  $\psi \triangleq \mathbf{A}\mathbf{f} \in \mathbb{R}^{CN}$  will have entries equal to  $f_{i}^{y_{i}} - f_{i}^{c}$  for each unique combination of  $c$  and  $i$ , of which there are  $CN$  in total. The OVE likelihood can now be written as  $p^{\mathrm{OVE}}(\mathbf{Y}|\psi) = 2^{N}\prod_{j=1}^{NC}\sigma(\psi_{j})$ , where the  $2^{N}$  term arises from the  $N$  cases in which  $\psi_{j} = 0$  due to comparing the ground truth logit with itself.

Analogous to (2), the likelihood of  $\psi$  conditioned on  $\omega$  is proportional to a diagonal Gaussian:

$$
p (\mathbf {Y} | \boldsymbol {\psi}, \boldsymbol {\omega}) \propto \prod_ {j = 1} ^ {N C} e ^ {- \omega_ {j} \psi_ {j} ^ {2} / 2} e ^ {\kappa_ {j} \psi_ {j}} \propto \mathcal {N} \left(\boldsymbol {\Omega} ^ {- 1} \boldsymbol {\kappa} | \boldsymbol {\psi}, \boldsymbol {\Omega} ^ {- 1}\right), \tag {9}
$$

where  $\kappa_{j} = 1 / 2$  and  $\Omega = \mathrm{diag}(\omega)$ . By exploiting the fact that  $\psi = \mathbf{A}\mathbf{f}$ , we can express the likelihood in terms of  $\mathbf{f}$  and write down the conditional posterior as follows:

$$
p (\mathbf {f} | \mathbf {X}, \mathbf {Y}, \omega) \propto \mathcal {N} \left(\Omega^ {- 1} \kappa \mid \mathbf {A f}, \Omega^ {- 1}\right) \mathcal {N} (\mathbf {f} \mid \boldsymbol {\mu}, \mathbf {K}) \propto \mathcal {N} (\mathbf {f} | \tilde {\Sigma} \left(\mathbf {K} ^ {- 1} \boldsymbol {\mu} + \mathbf {A} ^ {\top} \boldsymbol {\kappa}\right), \tilde {\Sigma}), \tag {10}
$$

where  $\tilde{\Sigma} = (\mathbf{K}^{-1} + \mathbf{A}^{\top}\Omega \mathbf{A})^{-1}$ , which is an expression remarkably similar to (3). Analogous to (4), the conditional distribution over  $\omega$  given  $\mathbf{f}$  and the data becomes  $p(\omega |\mathbf{y},\mathbf{f}) = \mathrm{PG}(\omega |\mathbf{1},\mathbf{A}\mathbf{f})$ .

The primary computational bottleneck of posterior inference lies in sampling  $\mathbf{f}$  from (10). Since  $\tilde{\Sigma}$  is a  $CN\times CN$  matrix, a naive implementation has complexity  $\mathcal{O}(C^3 N^3)$ . By utilizing the matrix inversion lemma and Gaussian sampling techniques (Doucet, 2010), this can be brought down to  $\mathcal{O}(CN^3)$ .

# 4.3 LEARNING COVARIANCE HYPERPARAMETERS FOR FEW-SHOT CLASSIFICATION

We now describe how we apply OVE Pólya-gamma augmented GPs to few-shot classification. We assume the standard episodic few-shot setup in which one observes a labeled support set  $S = (\mathbf{X}, \mathbf{Y})$ . Predictions must then be made for a query example  $(\mathbf{x}_*, \mathbf{y}_*)$ . We consider a zero-mean GP prior over the class logits  $\mathbf{f}^c(\mathbf{x}) \sim \mathcal{GP}(\mathbf{0}, k_\theta(\mathbf{x}, \mathbf{x}'))$ , where  $\theta$  are learnable parameters of our covariance function. These could include traditional hyperparameters such as lengthscales or the weights of a deep neural network as in deep kernel learning (Wilson et al., 2016).

We consider two objectives for learning hyperparameters of the covariance function: the marginal likelihood (ML)  $p_{\theta}(\mathbf{Y}|\mathbf{X})$  and the predictive likelihood (PL)  $p_{\theta}(\mathbf{y}_*|\mathbf{x}_*,\mathbf{X},\mathbf{Y})$ . Marginal likelihood measures the likelihood of the hyperparameters given the observed data and is intuitively appealing from a Bayesian perspective. On the other hand, many standard FSC methods optimize for predictive likelihood on the query set (Vinyals et al., 2016; Finn et al., 2017; Snell et al., 2017). Both objectives marginalize over latent functions, thereby making full use of our Bayesian formulation.

The details of these objectives and how we compute gradients can be found in Section B. Our learning algorithm for both marginal and predictive likelihood may be found in Section C. Details of computing the posterior predictive distribution  $p(\mathbf{y}_*|\mathbf{x}_*,\mathbf{X},\mathbf{Y},\boldsymbol{\omega})$  may be found in Section D. Finally, details of our chosen "cosine" kernel may be found in Section F.

# 5 EXPERIMENTS

We compare classification accuracy and uncertainty quantification to representative baselines for several major approaches to FSC: fine-tuning, metric learning, gradient-based meta-learning, Bayesian neural network and GP-based classifiers. An overview of the baselines we used can be found in Section E.2. One of our aims is to compare methods based on uncertainty quantification. We therefore developed new benchmark evaluations and tasks: few-shot calibration, robustness, and out-of-episode detection. In order to empirically compare methods, we could not simply borrow the accuracy results from other papers, but instead needed to train each of these baselines ourselves. For all baselines except Bayesian MAML, ABML, and Logistic Softmax GP, we ran the code from

(Patacchiola et al., 2019) and verified that the accuracies matched closely to those reported by (Patacchiola et al., 2019). Additional experimental details may be found in Section E. Code for our experiments may be found in the supplementary materials.

# 5.1 CLASSIFICATION ON FEW-SHOT BENCHMARKS

As mentioned above, we follow the training and evaluation protocol of Patacchiola et al. (2019) for this section. We train both 1-shot and 5-shot versions of our model in four different settings: Caltech-UCSD Birds (CUB) (Wah et al., 2011), mini-Imagenet with the split proposed by Ravi & Larochelle (2017), as well as two cross-domain transfer tasks: training on mini-ImageNet and testing on CUB, and from Omniglot (Lake et al., 2011) to EMNIST (Cohen et al., 2017). We employ the commonly-used Conv4 architecture with 64 channels (Vinyals et al., 2016) for all experiments. Further experimental details and comparisons across methods can be found in the appendix. Classification results are shown in Table 1 and 2. We find that our proposed Polya-Gamma OVE GPs yield strong classification results.

Table 1: Average accuracy and standard deviation (percentage) on 5-way FSC. Baseline results (through GPNet + Linear) are from Patacchiola et al. (2019). Evaluation is performed on 3,000 randomly generated test episodes. Standard deviation for our approach is computed by averaging over 5 batches of 600 episodes with different random seeds. The best results are highlighted in bold.  

<table><tr><td rowspan="2">Method</td><td colspan="2">CUB</td><td colspan="2">mini-ImageNet</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>Feature Transfer</td><td>46.19 ± 0.64</td><td>68.40 ± 0.79</td><td>39.51 ± 0.23</td><td>60.51 ± 0.55</td></tr><tr><td>Baseline++</td><td>61.75 ± 0.95</td><td>78.51 ± 0.59</td><td>47.15 ± 0.49</td><td>66.18 ± 0.18</td></tr><tr><td>MatchingNet</td><td>60.19 ± 1.02</td><td>75.11 ± 0.35</td><td>48.25 ± 0.65</td><td>62.71 ± 0.44</td></tr><tr><td>ProtoNet</td><td>52.52 ± 1.90</td><td>75.93 ± 0.46</td><td>44.19 ± 1.30</td><td>64.07 ± 0.65</td></tr><tr><td>RelationNet</td><td>62.52 ± 0.34</td><td>78.22 ± 0.07</td><td>48.76 ± 0.17</td><td>64.20 ± 0.28</td></tr><tr><td>MAML</td><td>56.11 ± 0.69</td><td>74.84 ± 0.62</td><td>45.39 ± 0.49</td><td>61.58 ± 0.53</td></tr><tr><td>GPNet + Linear</td><td>60.23 ± 0.76</td><td>74.74 ± 0.22</td><td>48.44 ± 0.36</td><td>62.88 ± 0.46</td></tr><tr><td>Bayesian MAML</td><td>55.93 ± 0.71</td><td>72.87 ± 0.26</td><td>44.46 ± 0.30</td><td>62.60 ± 0.25</td></tr><tr><td>Bayesian MAML (Chaser)</td><td>53.93 ± 0.72</td><td>71.16 ± 0.32</td><td>43.74 ± 0.46</td><td>59.23 ± 0.34</td></tr><tr><td>ABML</td><td>49.57 ± 0.42</td><td>68.94 ± 0.16</td><td>37.65 ± 0.22</td><td>56.08 ± 0.29</td></tr><tr><td>LSM GP + Cosine (ML)</td><td>60.23 ± 0.54</td><td>74.58 ± 0.25</td><td>46.75 ± 0.20</td><td>59.93 ± 0.31</td></tr><tr><td>LSM GP + Cosine (PL)</td><td>60.07 ± 0.29</td><td>78.14 ± 0.07</td><td>47.05 ± 0.20</td><td>66.01 ± 0.25</td></tr><tr><td>OVE PG GP + Cosine (ML) [ours]</td><td>63.98 ± 0.43</td><td>77.44 ± 0.18</td><td>50.02 ± 0.35</td><td>64.58 ± 0.31</td></tr><tr><td>OVE PG GP + Cosine (PL) [ours]</td><td>60.11 ± 0.26</td><td>79.07 ± 0.05</td><td>48.00 ± 0.24</td><td>67.14 ± 0.23</td></tr></table>

# 5.2 UNCERTAINTY QUANTIFICATION THROUGH CALIBRATION

We next turn to uncertainty quantification, an important concern for few-shot classifiers. When used in safety-critical applications such as medical diagnosis, it is important for a machine learning system to defer when there is not enough evidence to make a decision. Even in non-critical applications, precise uncertainty quantification helps practitioners in the few-shot setting determine when a class has an adequate amount of labeled data or when more labels are required, and can facilitate active learning.

We chose several commonly used metrics for calibration. Expected calibration error (ECE) (Guo et al., 2017) measures the expected binned difference between confidence and accuracy. Maximum calibration error (MCE) is similar to ECE but measures maximum difference instead of expected difference. Brier score (BRI) (Brier, 1950) is a proper scoring rule computed as the squared error between the output probabilities and the one-hot label. For a recent perspective on metrics for uncertainty evaluation, please refer to Ovadia et al. (2019). The results for representative approaches on 5-shot, 5-way CUB can be found in Figure 1. Our OVE PG GPs are the best calibrated overall across the metrics.

Table 2: Average accuracy and standard deviation (percentage) on 5-way cross-domain FSC, with the same experimental setup as in Table 1. Baseline results (through GPNet + Linear) are from (Patacchiola et al., 2019).  

<table><tr><td rowspan="2">Method</td><td colspan="2">Omniglot→EMNIST</td><td colspan="2">mini-ImageNet→CUB</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>Feature Transfer</td><td>64.22 ± 1.24</td><td>86.10 ± 0.84</td><td>32.77 ± 0.35</td><td>50.34 ± 0.27</td></tr><tr><td>Baseline++</td><td>56.84 ± 0.91</td><td>80.01 ± 0.92</td><td>39.19 ± 0.12</td><td>57.31 ± 0.11</td></tr><tr><td>MatchingNet</td><td>75.01 ± 2.09</td><td>87.41 ± 1.79</td><td>36.98 ± 0.06</td><td>50.72 ± 0.36</td></tr><tr><td>ProtoNet</td><td>72.04 ± 0.82</td><td>87.22 ± 1.01</td><td>33.27 ± 1.09</td><td>52.16 ± 0.17</td></tr><tr><td>RelationNet</td><td>75.62 ± 1.00</td><td>87.84 ± 0.27</td><td>37.13 ± 0.20</td><td>51.76 ± 1.48</td></tr><tr><td>MAML</td><td>72.68 ± 1.85</td><td>83.54 ± 1.79</td><td>34.01 ± 1.25</td><td>48.83 ± 0.62</td></tr><tr><td>GPNet + Linear</td><td>75.97 ± 0.70</td><td>89.51 ± 0.44</td><td>38.72 ± 0.42</td><td>54.20 ± 0.37</td></tr><tr><td>Bayesian MAML</td><td>63.94 ± 0.47</td><td>65.26 ± 0.30</td><td>33.52 ± 0.36</td><td>51.35 ± 0.16</td></tr><tr><td>Bayesian MAML (Chaser)</td><td>55.04 ± 0.34</td><td>54.19 ± 0.32</td><td>36.22 ± 0.50</td><td>51.53 ± 0.43</td></tr><tr><td>ABML</td><td>76.37 ± 0.29</td><td>87.96 ± 0.28</td><td>29.35 ± 0.26</td><td>45.74 ± 0.33</td></tr><tr><td>LSM GP + Cosine (ML)</td><td>62.91 ± 0.49</td><td>83.80 ± 0.13</td><td>36.41 ± 0.18</td><td>50.33 ± 0.13</td></tr><tr><td>LSM GP + Cosine (PL)</td><td>70.70 ± 0.36</td><td>86.59 ± 0.15</td><td>36.73 ± 0.26</td><td>56.70 ± 0.31</td></tr><tr><td>OVE PG GP + Cosine (ML) [ours]</td><td>68.43 ± 0.67</td><td>86.22 ± 0.20</td><td>39.66 ± 0.18</td><td>55.71 ± 0.31</td></tr><tr><td>OVE PG GP + Cosine (PL) [ours]</td><td>77.00 ± 0.50</td><td>87.52 ± 0.19</td><td>37.49 ± 0.11</td><td>57.23 ± 0.31</td></tr></table>

![](images/d7e19cb2094a4da57f759b1ed88636ea0b1dee58ad24708af349f5dde920c141.jpg)  
Figure 1: Reliability diagrams, expected calibration error (ECE), maximum calibration error (MCE), and Brier Score (BRI) for 5-shot 5-way tasks on CUB (additional calibration results can be found in the Section G). Metrics are computed on 3,000 random tasks from the test set.

# 5.3 ROBUSTNESS TO INPUT NOISE

Input examples for novel classes in FSC may have been collected under conditions that do not match those observed at training time. For example, labeled support images in a medical diagnosis application may come from a different hospital than the training set. To mimic a simplified version of this scenario, we investigate robustness to input noise. We used the Imagecorruptions package (Michaelis et al., 2019) to apply Gaussian noise, impulse noise, and defocus blur to both the support set and query sets of episodes at test time and evaluated both accuracy and calibration. We used corruption severity of 5 (severe) and evaluated across 1,000 randomly generated tasks on the three datasets involving natural images. The results are shown in Figure 2. We find that in general Bayesian approaches tend to be robust due to their ability to marginalize over hypotheses consistent with the support labels. Our approach is one of the top performing methods across all settings.

# 5.4 OUT-OF-EPISODE DETECTION

Finally, we measure performance on out-of-episode detection, another application in which uncertainty quantification is important. In this experiment, we used 5-way, 5-shot support sets at test time but incorporated out-of-episode examples into the query set. Each episode had 150 query examples: 15 from each of 5 randomly chosen in-episode classes and 15 from each of 5 randomly chosen out-of-episode classes. We then computed the AUROC of binary outlier detection using the negative of

![](images/8208c2713a6d044cfa741c024a5c8fb6aa26e3f7c0734a2ecb8e22b2b850968f.jpg)  
Figure 2: Accuracy  $(\uparrow)$  and Brier Score  $(\downarrow)$  when corrupting both support and query with noise on 5-way 5-shot tasks. Quantitative results may be found in Section H.

the maximum logit as the score. Intuitively, if none of the support classes assign a high logit to the example, it can be classified as an outlier. The results are shown in Figure 3. Our approach generally performs the best across the datasets.

![](images/a62689e92550efe6d2819e108fb02e16a5d97466aba84d64843752277748dbb9.jpg)  
Figure 3: Average AUROC  $(\uparrow)$  for out-of-episode detection. The AUC is computed separately for each episode and averaged across 1,000 episodes. Bars indicate a  $95\%$  bootstrapped conf. interval.

# 6 DISCUSSION

In our experiments, we observed that the fine-tuning approaches are strong baselines in terms of accuracy (Baseline++ in particular), but do not produce reliable estimates of uncertainty. Methods relying on Gaussian likelihoods (RelationNet and GPNet) tend to also exhibit poor uncertainty quantification. We hypothesize this is due to the ill-suited nature of applying Gaussian likelihoods to the fundamentally discrete task of classification. Optimizing for predictive cross-entropy generally improves classification accuracy and to some extent can remedy calibration issues of marginal likelihood-based methods. The OVE likelihood is better suited to classification than the Logistic Softmax likelihood, as can be seen by comparing the accuracy and calibration results of the ML versions of these models. Overall, our proposed OVE PG GP demonstrates strong performance across a wide range of scenarios.

# 7 CONCLUSION

In this work, we have proposed a Bayesian few-shot classification approach based on Gaussian processes. Our method replaces the ordinary softmax likelihood with a one-vs-each likelihood and applies Pólya-Gamma augmentation to perform inference. This allows us to model class logits directly as function values and efficiently marginalize over uncertainty in each few-shot episode. Modeling functions directly enables our approach to avoid the dependence on model size that posterior inference in weight-space based models inherently have. Our approach compares favorably to baseline FSC methods under a variety of dataset and shot configurations, including dataset transfer. We also demonstrate strong uncertainty quantification, robustness to input noise, and out-of-episode detection. We believe that Bayesian modeling in general and GPs in particular are powerful tools for handling uncertainty and hope that our work will lead to broader adoption of efficient Bayesian inference in the few-shot scenario.

# REFERENCES

James H Albert and Siddhartha Chib. Bayesian analysis of binary and polychotomous response data. Journal of the American statistical Association, 88(422):669-679, 1993.  
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. In International Conference on Machine Learning, 2015.  
Glenn W Brier. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1):1-3, 1950.  
Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Wang, and Jia-Bin Huang. A closer look at few-shot classification. In International Conference on Learning Representations, 2019.  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emmist: Extending mnist to handwritten letters. In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 2921-2926. IEEE, 2017.  
Randal Douc, Eric Moulines, and David Stoffer. Nonlinear time series: Theory, methods and applications with  $R$  examples. CRC press, 2014.  
A Doucet. A note on efficient conditional simulation of gaussian distributions. Departments of Computer Science and Statistics, University of British Columbia, 1020, 2010.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, 2017.  
Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning. In Advances in Neural Information Processing Systems, pp. 9516–9527, 2018.  
Théo Galy-Fajou, Florian Wenzel, Christian Donner, and Manfred Opper. Multi-class gaussian process classification made conjugate: Efficient inference via data augmentation. In Uncertainty in Artificial Intelligence, 2019.  
Mark Girolami and Simon Rogers. Variational bayesian multinomial probit regression with gaussian process priors. Neural Computation, 18(8):1790-1817, 2006.  
Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, and Richard E Turner. Meta-learning probabilistic inference for prediction. In International Conference on Learning Representations, 2019.  
Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Thomas Griffiths. Recasting gradient-based meta-learning as hierarchical bayes. arXiv preprint arXiv:1801.08930, 2018.  
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1321-1330. JMLR.org, 2017.  
Nathan Hilliard, Lawrence Phillips, Scott Howland, Artem Yankov, Courtney D Corley, and Nathan O Hadas. Few-shot learning with metric-agnostic conditional embeddings. arXiv preprint arXiv:1802.04376, 2018.  
Hyun-Chul Kim and Zoubin Ghahramani. Bayesian gaussian process classification with the em-ep algorithm. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(12):1948-1959, 2006.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Durk P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in neural information processing systems, pp. 2575-2583, 2015.  
Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese neural networks for one-shot image recognition. In ICML deep learning workshop, volume 2. Lille, 2015.

Brenden Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua Tenenbaum. One shot learning of simple visual concepts. In Proceedings of the annual meeting of the cognitive science society, 2011.  
Scott Linderman, Matthew J Johnson, and Ryan P Adams. Dependent multinomial models made easy: Stick-breaking with the polya-gamma augmentation. In Advances in Neural Information Processing Systems, pp. 3456-3464, 2015.  
Qiang Liu and Dilin Wang. Stein variational gradient descent: A general purpose bayesian inference algorithm. In Advances in neural information processing systems, pp. 2378-2386, 2016.  
Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, Alexander S. Ecker, Matthias Bethge, and Wieland Brendel. Benchmarking robustness in object detection: Autonomous driving when winter is coming. arXiv preprint arXiv:1907.07484, 2019.  
Thomas Peter Minka. A family of algorithms for approximate Bayesian inference. PhD thesis, Massachusetts Institute of Technology, 2001.  
Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, pp. 13969-13980, 2019.  
Massimiliano Patacchiola, Jack Turner, Elliot J Crowley, Michael O'Boyle, and Amos Storkey. Deep kernel transfer in gaussian processes for few-shot learning. arXiv preprint arXiv:1910.05199, 2019.  
Nicholas G Polson, James G Scott, and Jesse Windle. Bayesian inference for logistic models using polya-gamma latent variables. Journal of the American statistical Association, 108(504):1339-1349, 2013.  
Viraj Uday Prabhu. Few-shot learning for dermatological disease diagnosis. Master's thesis, Georgia Institute of Technology, 2019.  
Sachin Ravi and Alex Beatson. Amortized bayesian meta-learning. In International Conference on Learning Representations, 2019.  
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In International Conference on Learning Representations, 2017.  
Ryan Rifkin and Aldebaro Klautau. In defense of one-vs-all classification. Journal of machine learning research, 5(Jan):101-141, 2004.  
Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. arXiv preprint arXiv:1807.05960, 2018.  
Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, 2017.  
Shengyang Sun, Guodong Zhang, Jiaxin Shi, and Roger Grosse. Functional variational bayesian neural networks. In International Conference on Learning Representations, 2019.  
Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.  
Michalis Titsias. Variational learning of inducing variables in sparse gaussian processes. In Artificial Intelligence and Statistics, pp. 567-574, 2009.  
Michalis K Titsias. One-vs-each approximation to softmax for scalable estimation of probabilities. In Advances in Neural Information Processing Systems, pp. 4161-4169, 2016.  
Michalis K Titsias, Sotirios Nikoloutsopoulos, and Alexandre Galashov. Information theoretic meta learning with gaussian processes. arXiv preprint arXiv:2009.03228, 2020.

Prudencio Tossou, Basile Dura, Francois Laviolette, Mario Marchand, and Alexandre Lacoste. Adaptive deep kernel learning. arXiv preprint arXiv:1905.12131, 2019.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in Neural Information Processing Systems, 2016.  
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. computation & neural systems technical report. California Institute of Technology, 2011.  
Kuan-Chieh Wang, Jixuan Wang, Khai Truong, and Richard Zemel. Customizable facial gesture recognition for improved assistive technology. In ICLR AI for social good workshop, 2019.  
Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. Flipout: Efficient pseudo-independent weight perturbations on mini-batches. In International Conference on Representation Learning, 2018.  
Christopher KI Williams and Carl Edward Rasmussen. Gaussian processes for machine learning. MIT press Cambridge, MA, 2006.  
Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P Xing. Deep kernel learning. In Artificial Intelligence and Statistics, pp. 370-378, 2016.  
Jesse Windle, Nicholas G Polson, and James G Scott. Sampling polya-gamma random variates: alternate and approximate techniques. arXiv preprint arXiv:1405.0506, 2014.  
Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian model-agnostic meta-learning. In Advances in Neural Information Processing Systems, pp. 7332-7342, 2018.
