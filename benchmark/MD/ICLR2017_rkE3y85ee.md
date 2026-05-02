# CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX

Eric Jang

Google Brain

ejang@google.com

Shixiang Gu*

University of Cambridge  
MPI Tübingen

sg717@cam.ac.uk

Ben Poole*

Stanford University

poole@cs.stanford.edu

# ABSTRACT

Categorical variables are a natural choice for representing discrete structure in the world. However, stochastic neural networks rarely use categorical latent variables due to the inability to backpropagate through samples. In this work, we present an efficient gradient estimator that replaces the non-differentiable sample from a categorical distribution with a differentiable sample from a novel Gumbel-Softmax distribution. This distribution has the essential property that it can be smoothly annealed into a categorical distribution. We show that our Gumbel-Softmax estimator outperforms state-of-the-art gradient estimators on structured output prediction and unsupervised generative modeling tasks with categorical latent variables, and enables large speedups on semi-supervised classification.

# 1 INTRODUCTION

Stochastic neural networks with discrete random variables are a powerful technique for representing distributions encountered in unsupervised learning, language modeling, attention mechanisms, and reinforcement learning domains. For example, discrete variables have been used to learn probabilistic latent representations that correspond to distinct semantic classes (Kingma et al., 2014), image regions (Xu et al., 2015), and memory locations (Graves et al., 2014; Graves et al., 2016). Discrete representations are often more interpretable (Chen et al., 2016) and more computationally efficient (Rae et al., 2016) than their continuous analogues.

However, stochastic networks with discrete variables are difficult to train because the backpropagation algorithm — while permitting efficient computation of parameter gradients — cannot be applied to non-differentiable layers. Prior work on stochastic gradient estimation has traditionally focused on either score function estimators augmented with Monte Carlo variance reduction techniques (Paisley et al., 2012; Mnih & Gregor, 2014; Gu et al., 2016; Gregor et al., 2013), or biased path-derivative estimators for Bernoulli variables (Bengio et al., 2013). However, no existing gradient estimator has been formulated specifically for categorical variables. The contributions of this work are threefold:

1. We introduce Gumbel-Softmax, a continuous distribution on the simplex that can approximate categorical samples, and whose parameter gradients can be easily computed via the reparameterization trick.  
2. We show experimentally that Gumbel-Softmax outperforms all single-sample gradient estimators on both Bernoulli variables and categorical variables.  
3. We show that this estimator can be used to efficiently train semi-supervised models (e.g. Kingma et al. (2014)) without costly marginalization over unobserved categorical latent variables.

The practical outcome of this paper is a simple, differentiable sampling mechanism for categorical variables that can be integrated into neural networks and trained using standard backpropagation.

# 2 THE GUMBEL-SOFTMAX DISTRIBUTION

We begin by defining the Gumbel-Softmax distribution, a continuous distribution over the simplex that can approximate samples from a categorical distribution. Let  $z$  be a categorical variable with class probabilities  $\pi_1, \pi_2, \ldots, \pi_k$ . For the remainder of this paper we assume categorical samples are encoded as  $k$ -dimensional one-hot vectors lying on the corners of the  $(k - 1)$ -dimensional simplex,  $\Delta^{k - 1}$ . This allows us to define quantities such as the element-wise mean  $\mathbb{E}_p[z] = [\pi_1, \dots, \pi_k]$  of these vectors.

The Gumbel-Max trick (Gumbel, 1954; Maddison et al., 2014) provides a simple and efficient way to draw samples  $z$  from a categorical distribution with class probabilities  $\pi$ :

$$
z = \text {o n e} _ {\text {h o t}} \left(\underset {i} {\arg \max } \left[ g _ {i} + \log \pi_ {i} \right]\right) \tag {1}
$$

where  $g_{1}\ldots g_{k}$  are i.i.d samples drawn from Gumbel  $(0,1)^{1}$ . We use the softmax function as a continuous, differentiable approximation to arg max, and generate  $k$ -dimensional sample vectors  $y\in \Delta^{k - 1}$  where

$$
y _ {i} = \frac {\exp \left(\left(\log \left(\pi_ {i}\right) + g _ {i}\right) / \tau\right)}{\sum_ {j = 1} ^ {k} \exp \left(\left(\log \left(\pi_ {j}\right) + g _ {j}\right) / \tau\right)} \quad \text {f o r} i = 1, \dots , k. \tag {2}
$$

The density of the Gumbel-Softmax distribution (derived in Appendix B) is:

$$
p _ {\pi , \tau} \left(y _ {1}, \dots , y _ {k}\right) = \Gamma (k) \tau^ {k - 1} \left(\sum_ {i = 1} ^ {k} \pi_ {i} / y _ {i} ^ {\tau}\right) ^ {- k} \prod_ {i = 1} ^ {k} \left(\pi_ {i} / y _ {i} ^ {\tau + 1}\right) \tag {3}
$$

This distribution was independently discovered by Maddison et al. (2016), where it is referred to as the concrete distribution. As the softmax temperature  $\tau$  approaches 0, samples from the Gumbel-Softmax distribution become one-hot and the Gumbel-Softmax distribution becomes identical to the categorical distribution  $p(z)$ .

![](images/544870e8f6e8af826058b93ceffc59f9643d3a9e8f62dedc0dd63a3268ad7de7.jpg)  
Figure 1: The Gumbel-Softmax distribution interpolates between discrete one-hot-encoded categorical distributions and continuous categorical densities. (a) For low temperatures  $(\tau = 0.1, \tau = 0.5)$ , the expected value of a Gumbel-Softmax random variable approaches the expected value of a categorical random variable with the same logits. As the temperature increases  $(\tau = 1.0, \tau = 10.0)$ , the expected value converges to a uniform distribution over the categories. (b) Samples from Gumbel-Softmax distributions are identical to samples from a categorical distribution as  $\tau \to 0$ . At higher temperatures, Gumbel-Softmax samples are no longer one-hot, and become uniform as  $\tau \to \infty$ .

# 2.1 GUMBEL-SOFTMAX ESTIMATOR

The Gumbel-Softmax distribution is smooth for  $\tau > 0$ , and therefore has a well-defined gradient  $\frac{\partial y}{\partial \pi}$  with respect to the parameters  $\pi$ . Thus, by replacing categorical samples with Gumbel-Softmax samples we can use backpropagation to compute gradients (see Section 3.1). We denote

this procedure of replacing non-differentiable categorical samples with a differentiable approximation during training as the Gumbel-Softmax estimator.

While Gumbel-Softmax samples are differentiable, they are not identical to samples from the corresponding categorical distribution for non-zero temperature. For learning, there is a tradeoff between small temperatures, where samples are close to one-hot but the variance of the gradients is large, and large temperatures, where samples are smooth but the variance of the gradients is small (Figure 1). In practice, we start at a high temperature and anneal to a small but non-zero temperature.

In our experiments, we find that the softmax temperature  $\tau$  can be annealed according to a variety of schedules and still perform well. If  $\tau$  is a learned parameter (rather than annealed via a fixed schedule), this scheme can be interpreted as entropy regularization (Szegedy et al., 2015; Pereyra et al., 2016), where the Gumbel-Softmax distribution can adaptively adjust the "confidence" of proposed samples during the training process.

# 2.2 STRAIGHT-THROUGH GUMBEL- SOFTMAX ESTIMATOR

Continuous relaxations of one-hot vectors are suitable for problems such as learning hidden representations and sequence modeling. For scenarios in which we are constrained to sampling discrete values (e.g. from a discrete action space for reinforcement learning, or quantized compression), we discretize  $y$  using  $\arg \max$  but use our continuous approximation in the backward pass by approximating  $\nabla_{\theta}z\approx \nabla_{\theta}y$ . We call this the Straight-Through (ST) Gumbel Estimator, as it is reminiscent of the biased path-derivative estimator described in Bengio et al. (2013). ST Gumbel-Softmax allows samples to be sparse even when the temperature  $\tau$  is high.

# 3 RELATED WORK

In this section we review existing stochastic gradient estimation techniques for discrete variables (illustrated in Figure 2). Consider a stochastic computation graph (Schulman et al., 2015) with discrete random variable  $z$  whose distribution depends on parameter  $\theta$ , and cost function  $f(z)$ . The objective is to minimize the expected cost  $L(\theta) = \mathbb{E}_{z \sim p_{\theta}(z)}[f(z)]$  via gradient descent, which requires us to estimate  $\nabla_{\theta} \mathbb{E}_{z \sim p_{\theta}(z)}[f(z)]$ .

# 3.1 PATH DERIVATIVE GRADIENT ESTIMATORS

For distributions that are reparameterizable, we can compute the sample  $z$  as a deterministic function  $g$  of the parameters  $\theta$  and an independent random variable  $\epsilon$ , so that  $z = g(\theta, \epsilon)$ . The path-wise gradients from  $f$  to  $\theta$  can then be computed without encountering any stochastic nodes:

$$
\frac {\partial}{\partial \theta} \mathbb {E} _ {z \sim p _ {\theta}} [ f (z)) ] = \frac {\partial}{\partial \theta} \mathbb {E} _ {\epsilon} [ f (g (\theta , \epsilon)) ] = \mathbb {E} _ {\epsilon \sim p _ {\epsilon}} \left[ \frac {\partial f}{\partial g} \frac {\partial g}{\partial \theta} \right] \tag {4}
$$

For example, the normal distribution  $z \sim \mathcal{N}(\mu, \sigma)$  can be re-written as  $\mu + \sigma \cdot \mathcal{N}(0,1)$ , making it trivial to compute  $\partial z / \partial \mu$  and  $\partial z / \partial \sigma$ . This reparameterization trick is commonly applied to training variational autoencoders with continuous latent variables using backpropagation (Kingma & Welling, 2013; Rezende et al., 2014b). As shown in Figure 2, we exploit such a trick in the construction of the Gumbel-Softmax estimator.

Biased path derivative estimators can be utilized even when  $z$  is not reparameterizable. In general, we can approximate  $\nabla_{\theta}z\approx \nabla_{\theta}m(\theta)$ , where  $m$  is a differentiable proxy for the stochastic sample. For Bernoulli variables with mean parameter  $\theta$ , the Straight-Through (ST) estimator (Bengio et al., 2013) approximates  $m = \mu_{\theta}(z)$ , implying  $\nabla_{\theta}m = 1$ . For  $k = 2$  (Bernoulli), ST Gumbel-Softmax is similar to the slope-annealed Straight-Through estimator proposed by Chung et al. (2016), but uses a softmax instead of a hard sigmoid to determine the slope.

One limitation of the ST estimator is that backpropagating with respect to the sample-independent mean may cause discrepancies between the forward and backward pass, leading to higher variance. Gumbel-Softmax avoids this problem because each sample  $y$  is a differentiable proxy of the corresponding discrete sample  $z$ .

![](images/4e00935d3ebca30d30a83880840280e7dfa676e0fff45b679959f50db40376eb.jpg)  
Figure 2: Gradient estimation in stochastic computation graphs. (1)  $\nabla_{\theta}f(x)$  can be computed via backpropagation if  $x(\theta)$  is deterministic and differentiable. (2) The presence of stochastic node  $z$  precludes backpropagation as the sampler function does not have a well-defined gradient. (3) The score function estimator and its variants (NVIL, DARN, MuProp, VIMCO) obtain an unbiased estimate of  $\nabla_{\theta}f(x)$  by backpropagating along a surrogate loss  $\hat{f}\log p_{\theta}(z)$ , where  $\hat{f} = f(x) - b$  and  $b$  is a baseline for variance reduction. (4) The Straight-Through estimator, developed primarily for Bernoulli variables, approximates  $\nabla_{\theta}z\approx 1$ . (5) Gumbel-Softmax is a path derivative estimator for a continuous distribution  $y$  that approximates  $z$ . Reparameterization allows gradients to flow from  $f(y)$  to  $\theta$ .  $y$  can be annealed to one-hot categorical variables over the course of training.

![](images/da054f82d61533d11db83ea6d7c379ad568475f5cbedec9b7cbc7f072612f449.jpg)

![](images/d280fc26fa42aed9e51ccea552d09b6a4caa7fbfdf0404f9f4c63d5b85edb5bf.jpg)

![](images/335996e701bb175dba55ded40d89fa2faab7887ae8fc65d07b315bab4ebb70bc.jpg)

![](images/97f1956787f573c80a7229fa09ffd447cbf0d25e39b904b69cc074f3e4356b18.jpg)

# 3.2 SCORE FUNCTION-BASED GRADIENT ESTIMATORS

The score function estimator (SF, also referred to as REINFORCE (Williams, 1992) and likelihood ratio estimator (Glynn, 1990)) uses the identity  $\nabla_{\theta}\log p_{\theta}(z) = p_{\theta}(z)\nabla_{\theta}\log p_{\theta}(z)$  to derive the following unbiased estimator:

$$
\nabla_ {\theta} \mathbb {E} _ {z} [ f (z) ] = \mathbb {E} _ {z} [ f (z) \nabla_ {\theta} \log p _ {\theta} (z) ] \tag {5}
$$

SF only requires that  $p_{\theta}(z)$  is continuous in  $\theta$ , and does not require backpropagating through  $f$  or the sample  $z$ . However, SF suffers from high variance and is consequently slow to converge. In particular, the variance of SF scales linearly with the number of dimensions of the sample vector (Rezende et al., 2014a), making it especially challenging to use for categorical distributions.

The variance of a score function estimator can be reduced by subtracting a control variate  $b(z)$  from the learning signal  $f$ , and adding back its analytical expectation  $\mu_{b} = \mathbb{E}_{z}\left[b(z)\nabla_{\theta}\log p_{\theta}(z)\right]$  to keep the estimator unbiased:

$$
\begin{array}{l} \nabla_ {\theta} \mathbb {E} _ {z} [ f (z) ] = \mathbb {E} _ {z} [ f (z) \nabla_ {\theta} \log p _ {\theta} (z) + (b (z) \nabla_ {\theta} \log p _ {\theta} (z) - b (z) \nabla_ {\theta} \log p _ {\theta} (z)) ] (6) \\ = \mathbb {E} _ {z} \left[ (f (z) - b (z)) \nabla_ {\theta} \log p _ {\theta} (z) \right] + \mu_ {b} (7) \\ \end{array}
$$

We briefly summarize recent stochastic gradient estimators that utilize control variates. We direct the reader to Gu et al. (2016) for further detail on these techniques.

- NVIL (Mnih & Gregor, 2014) uses two baselines: (1) a moving average  $\bar{f}$  of  $f$  to center the learning signal, and (2) an input-dependent baseline computed by a 1-layer neural network fitted to  $f - \bar{f}$  (a control variate for the centered learning signal itself). Finally, variance normalization divides the learning signal by  $\max(1, \sigma_f)$ , where  $\sigma_f^2$  is a moving average of  $\operatorname{Var}[f]$ .

- DARN (Gregor et al., 2013) uses  $b = f(\bar{z}) + f'(\bar{z})(z - \bar{z})$ , where the baseline corresponds to the first-order Taylor approximation of  $f(z)$  from  $f(\bar{z})$ .  $\bar{z}$  is chosen to be  $1/2$  for Bernoulli variables, which makes the estimator biased for non-quadratic  $f$ , since it ignores the correction term  $\mu_b$  in the estimator expression.  
- MuProp (Gu et al., 2016) also models the baseline as a first-order Taylor expansion:  $b = f(\bar{z}) + f'(\bar{z})(z - \bar{z})$  and  $\mu_{b} = f'(\bar{z})\nabla_{\theta}\mathbb{E}_{z}[z]$ . To overcome backpropagation through discrete sampling, a mean-field approximation  $f_{MF}(\mu_{\theta}(z))$  is used in place of  $f(z)$  to compute the baseline and derive the relevant gradients.  
- VIMCO (Mnih & Rezende, 2016) is a gradient estimator for multi-sample objectives that uses the mean of other samples  $b = 1 / m \sum_{j \neq i} f(z_j)$  to construct a baseline for each sample  $z_i \in z_{1:m}$ . We exclude VIMCO from our experiments because we are comparing estimators for single-sample objectives, although Gumbel-Softmax can be easily extended to multi-sample objectives.

# 3.3 SEMI-SUPERVISED GENERATIVE MODELS

Semi-supervised learning considers the problem of learning from both labeled data  $(x,y)\sim \mathcal{D}_L$  and unlabeled data  $x\sim \mathcal{D}_U$ , where  $x$  are observations (i.e. images) and  $y$  are corresponding labels (e.g. semantic class). For semi-supervised classification, Kingma et al. (2014) propose a variational autoencoder (VAE) whose latent state is the joint distribution over a Gaussian "style" variable  $z$  and a categorical "semantic class" variable  $y$  (Figure 6, Appendix). The VAE objective trains a discriminative network  $q_{\phi}(y|x)$ , inference network  $q_{\phi}(z|x,y)$ , and generative network  $p_{\theta}(x|y,z)$  end-to-end by maximizing a variational lower bound on the log-likelihood of the observation under the generative model. For labeled data, the class  $y$  is observed, so inference is only done on  $z\sim q(z|x,y)$ . The variational lower bound on labeled data is given by:

$$
\log p _ {\theta} (x, y) \geq - \mathcal {L} (x, y) = \mathbb {E} _ {z \sim q _ {\phi} (z | x, y)} [ \log p _ {\theta} (x | y, z) ] - K L [ q (z | x, y) | | p _ {\theta} (y) p (z) ] \tag {8}
$$

For unlabeled data, difficulties arise because the categorical distribution is not reparameterizable. Kingma et al. (2014) approach this by marginalizing out  $y$  over all classes, so that for unlabeled data, inference is still on  $q_{\phi}(z|x,y)$  for each  $y$ . The lower bound on unlabeled data is:

$$
\begin{array}{l} \log p _ {\theta} (x) \geq - \mathcal {U} (x) = \mathbb {E} _ {z \sim q _ {\phi} (y, z | x)} [ \log p _ {\theta} (x | y, z) + \log p _ {\theta} (y) + \log p (z) - q _ {\phi} (y, z | x) ] (9) \\ = \sum_ {y} q _ {\phi} (y | x) (- \mathcal {L} (x, y) + \mathcal {H} (q _ {\phi} (y | x))) (10) \\ \end{array}
$$

The full maximization objective is:

$$
\mathcal {J} = \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {L}} [ - \mathcal {L} (x, y) ] + \mathbb {E} _ {x \sim \mathcal {D} _ {U}} [ - \mathcal {U} (x) ] + \alpha \cdot \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {L}} [ \log q _ {\phi} (y | x) ] \tag {11}
$$

where  $\alpha$  is the scalar trade-off between the generative and discriminative objectives.

One limitation of this approach is that marginalization over all  $k$  class values becomes prohibitively expensive for models with a large number of classes. If  $D, I, G$  are the computational cost of sampling from  $q_{\phi}(y|x)$ ,  $q_{\phi}(z|x, y)$ , and  $p_{\theta}(x|y, z)$  respectively, then training the unsupervised objective requires  $\mathcal{O}(D + k(I + G))$  for each forward/backward step. In contrast, Gumbel-Softmax allows us to backpropagate through  $y \sim q_{\phi}(y|x)$  for single sample gradient estimation, and achieves a cost of  $\mathcal{O}(D + I + G)$  per training step. Experimental comparisons in training speed are shown in Figure 5.

# 4 EXPERIMENTAL RESULTS

In our first set of experiments, we compare Gumbel-Softmax and ST Gumbel-Softmax to other stochastic gradient estimators: Score-Function (SF), DARN, MuProp, Straight-Through (ST), and Slope-Annealed ST. Each estimator is evaluated on two tasks: (1) structured output prediction and (2) variational training of generative models. We use the MNIST dataset with fixed binarization

for training and evaluation, which is common practice for evaluating stochastic gradient estimators (Salakhutdinov & Murray, 2008; Larochelle & Murray, 2011).

Learning rates are chosen from  $\{3\mathrm{e} - 5,1\mathrm{e} - 5,3\mathrm{e} - 4,1\mathrm{e} - 4,3\mathrm{e} - 3,1\mathrm{e} - 3\}$ ; we select the best learning rate for each estimator using the MNIST validation set, and report performance on the test set. Samples drawn from the Gumbel-Softmax distribution are continuous during training, but are discretized to one-hot vectors during evaluation. We also found that variance normalization was necessary to obtain competitive performance for SF, DARN, and MuProp. We used sigmoid activation functions for binary (Bernoulli) neural networks and softmax activations for categorical variables. Models were trained using stochastic gradient descent with momentum 0.9.

# 4.1 STRUCTURED OUTPUT PREDICTION WITH STOCHASTIC BINARY NETWORKS

The objective of structured output prediction is to predict the lower half of a  $28 \times 28$  MNIST digit given the top half of the image ( $14 \times 28$ ). This is a common benchmark for training stochastic binary networks (SBN) (Raiko et al., 2014; Gu et al., 2016; Mnih & Rezende, 2016). The minimization objective for this conditional generative model is an importance-sampled estimate of the likelihood objective,  $\mathbb{E}_{h \sim p_{\theta}(h_i | x_{\text{upper}})} \left[ \frac{1}{m} \sum_{i=1}^{m} \log p_{\theta}(x_{\text{lower}} | h_i) \right]$ , where  $m = 1$  is used for training and  $m = 1000$  is used for evaluation.

We trained a SBN with two hidden layers of 200 units each. This corresponds to either 200 Bernoulli variables (denoted as 392-200-200-392) or 20 categorical variables (each with 10 classes) with binarized activations (denoted as  $392 - (20 \times 10) - (20 \times 10) - 392$ ).

As shown in Figure 3, ST Gumbel-Softmax is on par with the other estimators for Bernoulli variables and outperforms on categorical variables. Meanwhile, Gumbel-Softmax outperforms other estimators on both Bernoulli and Categorical variables. We found that it was not necessary to anneal the softmax temperature for this task, and used a fixed  $\tau = 1$ .

![](images/f49f497964a51badc906fab4013d10dd1cd1884f2cee466308047b54220ea244.jpg)  
(a)

![](images/7b70102277163247b9342def51b40085ad40bc7cea7d678e6aef6298bae39fbb.jpg)  
(b)  
Figure 3: Test loss (negative log-likelihood) on the structured output prediction task with binarized MNIST using a stochastic binary network with (a) Bernoulli latent variables (392-200-200-392) and (b) categorical latent variables  $(392 - (20\times 10) - (20\times 10) - 392)$ .

# 4.2 GENERATIVE MODELING WITH VARIATIONAL AUTOENCODERS

We train variational autoencoders (Kingma & Welling, 2013), where the objective is to learn a generative model of binary MNIST images. In our experiments, we modeled the latent variable as a single hidden layer with 200 Bernoulli variables or 20 categorical variables  $(20 \times 10)$ . We use a uniform categorical prior rather than a Gumbel-Softmax prior in the training objective. Thus, the minimization objective during training is no longer a variational bound if the samples are not discrete. In practice, we find that optimizing this objective in combination with temperature annealing still min

imizes actual variational bounds on validation and test sets. Like the structured output prediction task, we use a multi-sample bound for evaluation with  $m = 1000$ .

The temperature is annealed using the schedule  $\tau = \max(0.5, \exp(-rt))$  of the global training step  $t$ , where  $\tau$  is updated every  $N$  steps.  $N \in \{500, 1000\}$  and  $r \in \{1\mathrm{e} - 5, 1\mathrm{e} - 4\}$  are hyperparameters for which we select the best-performing estimator on the validation set and report test performance.

As shown in Figure 4, ST Gumbel-Softmax outperforms other estimators for Categorical variables, and Gumbel-Softmax drastically outperforms other estimators in both Bernoulli and Categorical variables.

![](images/425f620908f58d8a17b01305dd3f13b5d2f5df378d0e1093ac27a08be9b1de98.jpg)  
(a)

![](images/c5086a4661c3c812825f35026457c160bc465f608770074b94f11d54ea465a20.jpg)  
(b)  
Figure 4: Test loss (negative variational lower bound) on binarized MNIST VAE with (a) Bernoulli latent variables  $(784 - 200 - 784)$  and (b) categorical latent variables  $(784 - (20 \times 10) - 200)$ .

Table 1: The Gumbel-Softmax estimator outperforms other estimators on Bernoulli and Categorical latent variables. For the structured output prediction (SBN) task, numbers correspond to negative log-likelihoods (nats) of input images (lower is better). For the VAE task, numbers correspond to negative variational lower bounds (nats) on the log-likelihood (lower is better).  

<table><tr><td></td><td>SF</td><td>DARN</td><td>MuProp</td><td>ST</td><td>Annealed ST</td><td>Gumbel-S.</td><td>ST Gumbel-S.</td></tr><tr><td>SBN (Bern.)</td><td>72.0</td><td>59.7</td><td>58.9</td><td>58.9</td><td>58.7</td><td>58.5</td><td>59.3</td></tr><tr><td>SBN (Cat.)</td><td>73.1</td><td>67.9</td><td>63.0</td><td>61.8</td><td>61.1</td><td>59.0</td><td>59.7</td></tr><tr><td>VAE (Bern.)</td><td>112.2</td><td>110.9</td><td>109.7</td><td>116.0</td><td>111.5</td><td>105.0</td><td>111.5</td></tr><tr><td>VAE (Cat.)</td><td>110.6</td><td>128.8</td><td>107.0</td><td>110.9</td><td>107.8</td><td>101.5</td><td>107.8</td></tr></table>

# 4.3 GENERATIVE SEMI-SUPERVISED CLASSIFICATION

We apply the Gumbel-Softmax estimator to semi-supervised classification on the binary MNIST dataset. We compare the original marginalization-based inference approach (Kingma et al., 2014) to single-sample inference with Gumbel-Softmax and ST Gumbel-Softmax.

We trained on a dataset consisting of 100 labeled examples (distributed evenly among each of the 10 classes) and 50,000 unlabeled examples, with dynamic binarization of the unlabeled examples for each minibatch. The discriminative model  $q_{\phi}(y|x)$  and inference model  $q_{\phi}(z|x,y)$  are each implemented as 3-layer convolutional neural networks with ReLU activation functions. The generative model  $p_{\theta}(x|y,z)$  is a 4-layer convolutional-transpose network with ReLU activations. Experimental details are provided in Appendix A.

Estimators were trained and evaluated against several values of  $\alpha = \{0.1, 0.2, 0.3, 0.8, 1.0\}$  and the best unlabeled classification results for test sets were selected for each estimator and reported

in Table 2. We used an annealing schedule of  $\tau = \max (0.5,\exp (-3\mathrm{e} - 5\cdot t))$ , updated every 2000 steps.

In Kingma et al. (2014), inference over the latent state is done by marginalizing out  $y$  and using the reparameterization trick for sampling from  $q_{\phi}(z|x,y)$ . However, this approach has a computational cost that scales linearly with the number of classes. Gumbel-Softmax allows us to backpropagate directly through single samples from the joint  $q_{\phi}(y,z|x)$ , achieving drastic speedups in training without compromising generative or classification performance. (Table 2, Figure 5).

Table 2: Marginalizing over  $y$  and single-sample variational inference perform equally well when applied to image classification on the binarized MNIST dataset (Larochelle & Murray, 2011). We report variational lower bounds and image classification accuracy for unlabeled data in the test set.  

<table><tr><td></td><td>ELBO</td><td>Accuracy</td></tr><tr><td>Marginalization</td><td>-106.8</td><td>92.6%</td></tr><tr><td>Gumbel</td><td>-109.6</td><td>92.4%</td></tr><tr><td>ST Gumbel-Softmax</td><td>-110.7</td><td>93.6%</td></tr></table>

In Figure 5, we show how Gumbel-Softmax versus marginalization scales with the number of categorical classes. For these experiments, we use MNIST images with randomly generated labels. Training the model with the Gumbel-Softmax estimator is  $2 \times$  as fast for 10 classes and  $9.9 \times$  as fast for 100 classes.

![](images/18e738369d82d47426c121f0733e68d629eb50023afd9b6fb7faf3effa4cd0c4.jpg)  
(a)

![](images/7b9f1057557de99c5353486bbe232c8924d18ab34ef5ab76adbffd4c07232d77.jpg)  
(b)  
Figure 5: Gumbel-Softmax allows us to backpropagate through samples from the posterior  $q_{\phi}(y|x)$ , providing a scalable method for semi-supervised learning for tasks with a large number of classes. (a) Comparison of training speed (steps/sec) between Gumbel-Softmax and marginalization (Kingma et al., 2014) on a semi-supervised VAE. Evaluations were performed on a GTX Titan X® GPU. (b) Visualization of MNIST analogies generated by varying style variable  $z$  across each row and class variable  $y$  across each column.

# 5 DISCUSSION

The primary contribution of this work is the reparameterizable Gumbel-Softmax distribution, whose corresponding estimator affords low-variance path-derivative gradients for the categorical distribution. We show that Gumbel-Softmax and Straight-Through Gumbel-Softmax are effective on structured output prediction and variational autoencoder tasks, outperforming existing stochastic gradient estimators for both Bernoulli and categorical latent variables. Finally, Gumbel-Softmax enables dramatic speedups in inference over discrete latent variables.

# ACKNOWLEDGMENTS

We sincerely thank Luke Vilnis, Vincent Vanhoucke, Luke Metz, David Ha, Laurent Dinh and Subhaneil Lahiri for helpful discussions and feedback.

# REFERENCES

Y. Bengio, N. Léonard, and A. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Info-gan: Interpretable representation learning by information maximizing generative adversarial nets. CoRR, abs/1606.03657, 2016.  
J. Chung, S. Ahn, and Y. Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.  
P. W Glynn. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM, 33(10):75-84, 1990.  
A. Graves, G. Wayne, M. Reynolds, T. Harley, I. Danihelka, A. Grabska-Barwińska, S. G. Colmenarejo, E. Grefenstette, T. Ramalho, J. Agapiou, et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626):471-476, 2016.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. CoRR, abs/1410.5401, 2014.  
K. Gregor, I. Danihelka, A. Mnih, C. Blundell, and D. Wierstra. Deep autoregressive networks. arXiv preprint arXiv:1310.8499, 2013.  
S. Gu, S. Levine, I. Sutskever, and A Mnih. MuProp: Unbiased Backpropagation for Stochastic Neural Networks. *ICLR*, 2016.  
E. J. Gumbel. Statistical theory of extreme values and some practical applications: a series of lectures. Number 33. US Govt. Print. Office, 1954.  
D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
D. P. Kingma, S. Mohamed, D. J. Rezende, and M. Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, pp. 3581-3589, 2014.  
H. Larochelle and I. Murray. The neural autoregressive distribution estimator. In AISTATS, volume 1, pp. 2, 2011.  
C. J. Maddison, D. Tarlow, and T. Minka. A* sampling. In Advances in Neural Information Processing Systems, pp. 3086-3094, 2014.  
C. J. Maddison, A. Mnih, and Y. Whye Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. ArXiv e-prints, November 2016.  
A. Mnih and K. Gregor. Neural variational inference and learning in belief networks. ICML, 31, 2014.  
A. Mnih and D. J. Rezende. Variational inference for monte carlo objectives. arXiv preprint arXiv:1602.06725, 2016.  
J. Paisley, D. Blei, and M. Jordan. Variational Bayesian Inference with Stochastic Search. *ArXiv e-prints*, June 2012.  
Gabriel Pereyra, Geoffrey Hinton, George Tucker, and Lukasz Kaiser. Regularizing neural networks by penalizing confident output distributions. 2016.  
J. W Rae, J. J Hunt, T. Harley, I. Danihelka, A. Senior, G. Wayne, A. Graves, and T. P Lillicrap. Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes. *ArXiv e-prints*, October 2016.  
T. Raiko, M. Berglund, G. Alain, and L. Dinh. Techniques for learning binary stochastic feedforward neural networks. arXiv preprint arXiv:1406.2989, 2014.

D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082, 2014a.  
D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of The 31st International Conference on Machine Learning, pp. 1278-1286, 2014b.  
R. Salakhutdinov and I. Murray. On the quantitative analysis of deep belief networks. In Proceedings of the 25th international conference on Machine learning, pp. 872-879. ACM, 2008.  
J. Schulman, N. Heess, T. Weber, and P. Abbeel. Gradient estimation using stochastic computation graphs. In Advances in Neural Information Processing Systems, pp. 3528-3536, 2015.  
C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015.  
R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.  
K. Xu, J. Ba, R. Kiros, K. Cho, A. C. Courville, R. Salakhutdinov, R. S. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. CoRR, abs/1502.03044, 2015.
