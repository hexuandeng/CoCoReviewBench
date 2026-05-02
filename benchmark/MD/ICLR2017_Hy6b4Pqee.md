# DEEP PROBABILISTIC PROGRAMMING

Dustin Tran

Columbia University

Matthew D. Hoffman

Adobe Research

Rif A. Saurous

Google Research

Eugene Brevdo

Google Brain

Kevin Murphy

Google Research

David M. Blei

Columbia University

# ABSTRACT

We propose Edward, a Turing-complete probabilistic programming language. Edward builds on two compositional representations—random variables and inference. By treating inference as a first class citizen, on a par with modeling, we show that probabilistic programming can be as flexible and computationally efficient as traditional deep learning. For flexibility, Edward makes it easy to fit the same model using a variety of composable inference methods, ranging from point estimation, to variational inference, to MCMC. In addition, Edward can reuse the modeling representation as part of inference, facilitating the design of rich variational models and generative adversarial networks. For efficiency, Edward is integrated into TensorFlow, providing significant speedups over existing probabilistic systems. For example, on a benchmark logistic regression task, Edward is at least 35x faster than Stan and PyMC3.

# 1 INTRODUCTION

The nature of deep neural networks is compositional. Users can connect layers in creative ways, without having to worry about how to perform testing (forward propagation) or inference (gradient-based optimization, with back propagation and automatic differentiation).

In this paper, we design compositional representations for probabilistic programming. Probabilistic programming lets users specify generative probabilistic models as programs and then "compile" those models down into inference procedures. Probabilistic models are also compositional in nature, and most previous work focuses on building rich probabilistic programs by composing random variables (Goodman et al., 2012; Ghahramani, 2015; Lake et al., 2016).

Less work, however, has considered an analogous compositionality for inference. Rather, most existing probabilistic programming languages treat the inference engine as a black box, abstracted away from the model. These cannot capture the recent advances in probabilistic inference that reuse the model's representation. For example, these advances have become important to variational inference (Kingma & Welling, 2014; Rezende & Mohamed, 2015; Tran et al., 2016b) and generative adversarial networks (Goodfellow et al., 2014).

We propose Edward<sup>1</sup>, a new Turing-complete probabilistic programming language which builds on two compositional representations—one for random variables and one for inference. We show how to integrate Edward into existing computational graph frameworks such as TensorFlow (Abadi et al., 2016). Frameworks like TensorFlow provide computational benefits like distributed training, parallelism, vectorization, and GPU support "for free." We also show how Edward makes it easy to fit the same model using a variety of composable inference methods, ranging from point estimation, to variational inference, to MCMC. By treating inference as a first class citizen, on a par with modeling, we show that probabilistic programming can be as computationally efficient and flexible as traditional deep learning. For example, our implementation of Hamiltonian Monte Carlo is  $35\mathrm{x}$  faster than existing software.

# 2 RELATED WORK

Probabilistic programming languages (PPLs) typically trade off the expressiveness of the language with the computational efficiency of inference. On one side, there are languages which emphasize expressiveness (Pfeffer, 2001; Milch et al., 2005; Pfeffer, 2009; Goodman et al., 2012), representing a rich class beyond graphical models. Each employs a generic inference engine, but scales poorly with respect to model and data size. On the other side, there are languages which emphasize efficiency (Spiegelhalter et al., 1995; Murphy, 2001; Plummer, 2003; Carpenter et al., 2016). The PPL is restricted to a specific class of models, and inference algorithms are optimized to be efficient for this class. For example, Infer.NET enables fast message passing for graphical models (Minka et al., 2014), and Augur enables data parallelism with GPUs for Gibbs sampling in Bayesian networks (Tristan et al., 2014). Edward bridges this gap. It is Turing complete—it supports any computable probability distribution—and it supports efficient algorithms, such as those that leverage model structure and those that scale to massive data.

There has been some prior research on efficient algorithms in Turing-complete languages. Venture and Anglican propose inference as a collection of local inference problems, defined over program fragments (Mansinghka et al., 2014; Wood et al., 2014). This produces fast program-specific inference code, which we build on. However, neither system supports inference methods such as programmable posterior approximations, inference models, or data subsampling. WebPPL does support amortized inference (Ritchie et al., 2016). However, its design does not allow reuse of random variables to construct a variational approximation; rather, it annotates the original program and leverages helper functions, which is a less flexible strategy. Finally, inference is defined as program transformations in Kiselyov & Shan (2009); Scibior et al. (2015); Zinkov & Shan (2016), where the output of inference can be composed as part of another program. Edward builds on this idea to compose not only inference within modeling but also modeling within inference (see Section 3.1).

# 3 COMPOSITIONAL REPRESENTATIONS FOR PROBABILISTIC MODELS

We first develop compositional representations for probabilistic models. These representations are designed to also be usable during inference.

In Edward, random variables are the key compositional representation. They are class objects with methods, for example, to compute the log density and to sample. Further, each random variable  $\mathbf{x}$  is associated to a tensor (multi-dimensional array)  $\mathbf{x}^*$ , which represents a single sample  $\mathbf{x}^* \sim p(\mathbf{x})$ . This association embeds the random variable into a computational graph, a symbolic framework where nodes represent operations on tensors and edges represent tensors communicated between them (Culler, 1986).

This design facilitates developing probabilistic programs in a computational graph framework. Importantly, all computation is represented on the graph. This makes it easy to compose random variables with complex deterministic structure such as deep neural networks, a diverse set of math operations, and third party libraries that build on the same framework. The design also enables compositions of random variables to capture complex stochastic structure.

As a simple example, we illustrate a Beta-Bernoulli model,  $p(\mathbf{x}, \theta) = \mathrm{Beta}(\theta | 1, 1) \prod_{n=1}^{50} \mathrm{Bernoulli}(x_n | \theta)$ , where  $\theta$  is a latent probability shared across the 50 data points  $\mathbf{x} \in \{0, 1\}^{50}$ . The random variable  $\mathbf{x}$  is 50-dimensional, parameterized by the random tensor  $\theta^*$ . Fetching the object  $\mathbf{x}$  runs the graph: it simulates from the generative process and outputs a binary vector of 50 elements.

![](images/b1273a453754f8fdf16203ba276f865a700f941b5cf8f3f603c7b38d5dcf1eb9.jpg)  
Figure 1: Beta-Bernoulli program (left) alongside its computational graph (right). Fetching  $\mathbf{x}$  from the graph generates a binary vector of 50 elements.

![](images/a6fdded8e2ce8d59c0963fb45b7cce031c1ddc6e3448ba89a131f57703e813e1.jpg)  
Figure 2: Variational auto-encoder for a data set of  $28 \times 28$  pixel images: (left) graphical model, with dotted lines for the inference model; (right) probabilistic program, with 2-layer neural networks.

All computation is registered symbolically on random variables and not over their execution. Symbolic representations do not require reifying the full model, which leads to unreasonable memory consumption for large models (Tristan et al., 2014). Moreover, it enables us to simplify both deterministic and stochastic operations in the graph, before executing any code (Scibior et al., 2015; Zinkov & Shan, 2016).

With computational graphs, it is also natural to build mutable states within the probabilistic program. As a typical use of computational graphs, such states can define model parameters; in TensorFlow, this is given by a tf.Variable. Another use case is for building discriminative models  $p(\mathbf{y} \mid \mathbf{x})$ , where  $\mathbf{x}$  are features that are input as training or test data. The program can be written independent of the data, using a mutable state (tf.placeholder) for  $\mathbf{x}$  in its graph. During training and testing, we feed the placeholder the appropriate values.

In Appendix A, we provide examples of a Bayesian neural network for classification (A.1), latent Dirichlet allocation (A.4), and Gaussian matrix factorization (A.5). We present others below.

# 3.1 EXAMPLE: VARIATIONAL AUTO-ENCODER

Figure 2 implements a variational auto-encoder (VAE) (Kingma & Welling, 2014; Rezende et al., 2014) in Edward. It comprises of a probabilistic model over data and a variational model over latent variables. Here we use random variables to construct both the probabilistic model and the variational model, which is fitted during inference (more details in Section 4).

There are  $N$  data points  $\{x_{n}\}$  and  $d$  latent variables per data point  $\{z_{n}\}$ . The program uses TensorFlow Slim (Guadarrama & Silberman, 2016) to define the neural networks. The probabilistic model is parameterized by a 2-layer neural network, with 256 hidden units (and ReLU activation), and generates  $28 \times 28$  pixel images. The variational model is parameterized by a 2-layer inference network, with 256 hidden units and outputs parameters of a normal posterior approximation.

The probabilistic program is concise. Core elements of the VAE—such as its distributional assumptions and neural net architectures—are all extensible. With model compositionality, we can embed it into more complicated models (Gregor et al., 2015; Rezende et al., 2016) and for other learning tasks (Kingma et al., 2014). With inference compositionality (which we discuss in Section 4), we can embed it into more complicated algorithms, such as with expressive variational approximations (Rezende & Mohamed, 2015; Tran et al., 2016b; Kingma et al., 2016) and alternative objectives (Ranganath et al., 2016a; Li & Turner, 2016; Dieng et al., 2016).

# 3.2 EXAMPLE: BAYESIAN RECURRENT NEURAL NETWORK WITH VARIABLE LENGTH

Random variables can also be composed with control flow operations. As an example, Figure 3 implements a Bayesian recurrent neural network (RNN) with variable length. The data is a sequence of inputs  $\{\mathbf{x}_1,\dots ,\mathbf{x}_T\}$  and outputs  $\{y_{1},\ldots ,y_{T}\}$  of length  $T$  with  $\mathbf{x}_t\in \mathbb{R}^D$  and  $y_{t}\in \mathbb{R}$  per time step. For  $t = 1,\dots ,T$ , a RNN applies the update

$$
\mathbf {h} _ {t} = \tanh  \left(\mathbf {W} _ {h} \mathbf {h} _ {t - 1} + \mathbf {W} _ {x} \mathbf {x} _ {t} + \mathbf {b} _ {h}\right)
$$

where the previous hidden state is  $\mathbf{h}_{t - 1}\in \mathbb{R}^H$ . We feed each hidden state into the output's likelihood,  $y_{t}\sim \mathrm{Normal}(\mathbf{W}_{y}\mathbf{h}_{t} + \mathbf{b}_{y},1)$ . We place a standard normal prior over all parameters

![](images/c1d89a8d7f6d68c085a0098f830c6f9d5d30fc98cdfc03fff046229cd2937a22.jpg)  
Figure 3: Bayesian RNN: (left) graphical model; (right) probabilistic program. The program has an unspecified number of time steps; it uses a symbolic for loop (tf.scan).

```python
1 def rnn_cell(hprev, xt):   
2 return tf.tanh(tf.dot(hprev, Wh) + tf.dot(xt, Wx) + bh)   
3   
4 Wh = Normal(mu=tf.zeros([H, H]), sigma=tf.ones([H, H]))   
5 Wx = Normal(mu=tf.zeros([D, H]), sigma=tf.ones([D, H]))   
6 Wy = Normal(mu=tf.zeros([H, 1]), sigma=tf.ones([H, 1]))   
7 bh = Normal(mu=tf.zeros(H), sigma=tf.ones(H))   
8 by = Normal(mu=tf.zeros(1), sigma=tf.ones(1))   
9   
10 x = tf.placeholder(tf.float32, [None, D])   
11 h = tf.scan(rnn_cell, x, initializer=tf.zeros(H))   
12 y = Normal(mu=tf/matmul(h, Wy) + by, sigma=1.0)
```

$\{\mathbf{W}_h \in \mathbb{R}^{H \times H}, \mathbf{W}_x \in \mathbb{R}^{D \times H}, \mathbf{W}_y \in \mathbb{R}^{H \times 1}, \mathbf{b}_h \in \mathbb{R}^H, \mathbf{b}_y \in \mathbb{R}\}$ . Our implementation is dynamic: it differs from a RNN with fixed length, which requires padding and unrolling the computation.

# 3.3 STOCHASTIC CONTROL FLOW AND MODEL PARALLELISM

![](images/d8ed55c69ce445bfc1e05e385ebc8ee2144f2c985a438ec16cfeafd93a6bf988.jpg)  
Figure 4: Computational graph for a probabilistic program with stochastic control flow.

Random variables can also be placed in the control flow itself, enabling probabilistic programs with stochastic control flow. Stochastic control flow defines dynamic conditional dependencies, known in the literature as contingent or existential dependencies (Mansinghka et al., 2014; Wu et al., 2016). See Figure 4, where  $\mathbf{x}$  may or may not depend on a for a given execution. In Appendix A.3, we use stochastic control flow to implement a Dirichlet process mixture model.

Stochastic control flow produces difficulties for algorithms that use the graph structure because the relationship of conditional dependencies changes across execution traces. The computational graph, however, provides an elegant way of teasing out static conditional dependence structure (p) from dynamic dependence structure (a). We can perform model parallelism (parallel computation across components of the model) over the static structure with GPUs and batch training. We can use more generic computations to handle the dynamic structure.

# 4 COMPOSITIONAL REPRESENTATIONS FOR INFERENCE

We have described random variables as a representation for building rich probabilistic programs over computational graphs. We now describe a compositional representation for inference.

In inference, we desire two criteria: (a) support for many classes of inference, where the form of the inferred posterior depends on the algorithm; and (b) invariance of inference under the computational graph, that is, the posterior can be further composed as part of another model.

To explain our approach, we will use a simple hierarchical model as a running example. Figure 5 shows a joint distribution  $p(\mathbf{x}, \mathbf{z}, \beta)$  of data  $\mathbf{x}$ , local variables  $\mathbf{z}$ , and global variables  $\beta$ . The ideas here extend to more expressive programs.

![](images/6abe1aeb1722f3538c21af43cd563acac491c0168f1ac55a3c83b2ac4064087f.jpg)  
Figure 5: Hierarchical model: (left) graphical model; (right) probabilistic program. It is a mixture of Gaussians over  $D$ -dimensional data  $\{x_{n}\} \in \mathbb{R}^{N\times D}$ . There are  $K$  latent cluster means  $\beta \in \mathbb{R}^{K\times D}$ .

1  $\mathrm{N} = 10000$  # number of data points  
2  $\mathrm{D} = 2$  # data dimension  
3  $\mathrm{K} = 5$  # number of clusters  
4  
5 beta  $=$  Normal(mu=tf.zeros([K,D]),sigma=tf.ones([K,D]))  
6 z  $=$  Categorical(logits=tf.zeros([N,K]))  
7 x  $=$  Normal(mu=tf.gather(beta,z),sigma=tf.ones([N,D]))

# 4.1 INFERENCE AS STOCHASTIC GRAPH OPTIMIZATION

Given data  $\mathbf{x}_{\mathrm{train}}$ , inference aims to calculate the posterior  $p(\mathbf{z},\beta \mid \mathbf{x}_{\mathrm{train}};\boldsymbol {\theta})$ , where  $\boldsymbol{\theta}$  are any model parameters that we will compute point estimates for. We formalize this as the following optimization problem:

$$
\min  _ {\boldsymbol {\lambda}, \boldsymbol {\theta}} \mathcal {L} (p (\mathbf {z}, \beta \mid \mathbf {x} _ {\text {t r a i n}}; \boldsymbol {\theta}), q (\mathbf {z}, \beta ; \boldsymbol {\lambda})), \tag {1}
$$

where  $q(\mathbf{z},\beta ;\boldsymbol {\lambda})$  is an approximation to the posterior  $p(\mathbf{z},\beta \mid \mathbf{x}_{\mathrm{train}})$ , and  $\mathcal{L}(\cdot)$  is a loss function with respect to  $p$  and  $q$ .

The choice of approximation  $q$ , loss  $\mathcal{L}$ , and rules to update parameters  $\{\theta, \lambda\}$  are specified by an inference algorithm. (Note  $q$  can be nonparametric, such as a point or a collection of samples.)

In Edward, we write this problem as follows:

```typescript
1 inference = ed.Inference({beta: qbeta, z: qz}, data={x: x_train})
```

Inference is an abstract class which takes two inputs. The first is a collection of latent random variables beta and z, along with "posterior variables" qbeta and qz, which are associated to their respective latent variables. The second is a collection of observed random variables x, which is associated to the data x_train.

The idea is that Inference defines and solves the optimization in Equation 1. It adjusts parameters of the distribution of qbeta and qz (and any model parameters) to be close to the posterior  $p(\mathbf{z},\beta \mid \mathbf{x}_{train})$ .

Class methods are available to control the inference. Calling inference.initialize() builds a computational graph to update  $\{\theta, \lambda\}$ . Calling inference.update() runs this computation once to update  $\{\theta, \lambda\}$ ; we call the method in a loop until convergence. Below we will derive subclasses of Inference to represent many inference algorithms.

# 4.2 CLASSES OF INFERENCE

Edward uses stochastic graph optimization to implement many algorithms. We illustrate several classes below: variational inference, Monte Carlo, and generative adversarial networks.

Variational inference posits a family of approximating distributions and finds the closest member in the family to the posterior (Jordan et al., 1999). In Edward, we build the variational family in the graph; see Figure 6 (left). The variational family has mutable variables representing its parameters  $\lambda = \{\pi, \mu, \sigma\}$ , where  $q(\beta; \mu, \sigma) = \mathrm{Normal}(\beta; \mu, \sigma)$  and  $q(\mathbf{z}; \pi) = \mathrm{Categorical}(\mathbf{z}; \pi)$ .

Specific variational algorithms inherit from the Variational Inference class. Each defines its own methods, such as a loss function and gradient. For example, we represent maximum a posteriori (MAP) estimation with an approximating family (qbeta and qz) of PointMass random variables, i.e., with all probability mass concentrated at a point. MAP inherits from Variational Inference and defines a loss function and update rules; it uses existing optimizers inside TensorFlow. In Section 5.1, we experiment with multiple gradient estimators for black box variational inference (Ranganath et al., 2014). Each estimator implements the same loss and a different update rule.

```python
1 qbeta = Normal( 1 T = 10000 # number of samples  
2 mu=tf.Variable(tf.zeros([K, D))), 2 qbeta = Empirical(  
3 sigma=tf.exp(tf.Variable(tf.zeros[K, D])) 3 params=tf.Variable(tf.zeros([T, K, D]))  
4 qz = Categorical( 4 qz = Empirical(  
5 logits=tf.Variable(tf.zeros[N, K])) 5 params=tf.Variable(tf.zeros([T, N]))  
6 6  
7 inference = ed.VariationalInference( 7 inference = ed.MonteCarlo(  
8 {beta: qbeta, z: qz}, data={x: x_train}) 8 {beta: qbeta, z: qz}, data={x: x_train})
```

![](images/16f6f201900451021a9708b5337e775c944b05f3c0f47b8b503294fdf9f578c8.jpg)  
Figure 6: (left) Variational inference. (right) Monte Carlo.  
Figure 7: Generative adversarial networks: (left) graphical model; (right) probabilistic program. The model (generator) is augmented with fake data and a discriminator for training.

```python
def generative_network(z):
    h = slimFULLConnected(z, 256, activation_fn=tf.nn.relu)
    return slimFULLConnected(h, 28 * 28, activation_fn=None)
def discriminative_network(x):
    h = slimFULLConnected(z, 28 * 28, activation_fn=tf.nn.relu)
    return slimFULLConnected(h, 1, activation_fn=None)
# Probabilistic model
z = Normal(mu=tf.zeros([M, d]), sigma=tf.ones([M, d]))
x = generative_network(z)
# augmentation for GAN-based inference
y Fake = Bernoulli(logits=discriminative_network(x))
y_real = Bernoulli(logits=discriminative_network(x_train))
data = {y_real: tf.ones(N), y_fake: tf.zeros(M)}
inference = ed.GANInference(data=data)
```

Monte Carlo approximates the posterior using samples (Robert & Casella, 1999). Monte Carlo is an inference where the approximating family is an empirical distribution,  $q(\beta; \{\beta^{(t)}\}) = \frac{1}{T} \sum_{t=1}^{T} \delta(\beta, \beta^{(t)})$  and  $q(\mathbf{z}; \{\mathbf{z}^{(t)}\}) = \frac{1}{T} \sum_{t=1}^{T} \delta(\mathbf{z}, \mathbf{z}^{(t)})$ . The parameters are  $\lambda = \{\beta^{(t)}, \mathbf{z}^{(t)}\}$ . See Figure 6 (right). Monte Carlo algorithms proceed by updating one sample  $\beta^{(t)}$ ,  $\mathbf{z}^{(t)}$  at a time in the empirical approximation. Specific MC samplers determine the update rules; they can use gradients such as in Hamiltonian Monte Carlo (Neal, 2011) and graph structure such as in sequential Monte Carlo (Doucet et al., 2001).

Edward also supports non-Bayesian methods such as generative adversarial networks (GANs) (Goodfellow et al., 2014). See Figure 7. The model (generator) has a standard normal prior  $\mathbf{z}$  over  $M$  data points, each with  $d$  latent dimensions; the hidden variable  $\mathbf{z}$  feeds into a generative_network function, a neural network that outputs real-valued data  $\mathbf{x}$ . Inference augments the model in a noise-contrastive setup (Gutmann & Hyvarinen, 2010). There is a discriminative_network which takes in (real or fake) data and outputs the probability that the data is real (in logit parameterization). We then build GANInference; running it optimizes parameters inside the two neural network functions. This approach applies to many GAN extensions (e.g., Denton et al. (2015); Li et al. (2015)).

Finally, this approach also extends to algorithms that usually require tedious algebraic manipulation. With symbolic algebra on the nodes of the computational graph, we can uncover conjugacy relationships between random variables. Users can then integrate out variables to automatically derive classical Gibbs (Gelfand & Smith, 1990), mean-field updates (Bishop, 2006), and exact inference.

# 4.3 COMPOSING INFERENCES

Core to Edward's design is that inference can be written as a collection of separate inference programs. Below we demonstrate variational EM, with an (approximate) E-step over local variables and an M-step over global variables. We alternate with one update of each (Neal & Hinton, 1993).

```txt
1 qbeta = PointMass(param=tf.Variable(tf.zeros([K, D]))  
2 qz = Categorical(logits=tf.Variable(tf.zeros[N, K]))
```

```python
3
4     inference_e = ed.VariationalInference({z: qz}, data={x: x_data, beta: qbeta})
5     inference_m = ed.MAP({beta: qbeta}, data={x: x_data, z: qz})
6
7     for _ in range(10000):
8         inference_e.update()
9         inference_m.update()
```

This extends to many other cases, such as exact EM for exponential families, contrastive divergence (Hinton, 2002), pseudo-marginal methods (Andrieu & Roberts, 2009), and Gibbs sampling within variational inference (Wang & Blei, 2012; Hoffman & Blei, 2015). We can also write message passing algorithms, which solve a collection of local inference problems (Koller & Friedman, 2009). For example, classical message passing uses exact local inference; expectation propagation locally minimizes  $\mathrm{KL}(p\parallel q)$  (Minka, 2001).

# 4.4 DATA SUBSAMPLING

Stochastic optimization (Bottou, 2010) scales inference to massive data and is key to algorithms such as stochastic gradient Langevin dynamics (Welling & Teh, 2011) and stochastic variational inference (Hoffman et al., 2013). The idea is to cheaply estimate the model's log joint density in an unbiased way. At each step, one subsamples a data set  $\{x_{m}\}$  of size  $M$  and then scales densities with respect to local variables,

$$
\begin{array}{l} \log p (\mathbf {x}, \mathbf {z}, \beta) = \log p (\beta) + \sum_ {n = 1} ^ {N} \left[ \log p (x _ {n} \mid z _ {n}, \beta) + \log p (z _ {n} \mid \beta) \right] \\ \approx \log p (\beta) + \frac {N}{M} \sum_ {m = 1} ^ {M} \left[ \log p (x _ {m} \mid z _ {m}, \beta) + \log p (z _ {m} \mid \beta) \right]. \\ \end{array}
$$

To support stochastic optimization, we represent only a subgraph of the full model; this prevents reifying the full model, which can lead to unreasonable memory consumption (Tristan et al., 2014). During initialization, we pass in a dictionary to properly scale the arguments.

![](images/5bda7a81af6a887aade5a39ae14abf93a4c1e52d3b199f3786492f80f7341012.jpg)  
Figure 8: Data subsampling with a hierarchical model. We define a subgraph of the full model, forming a plate of size  $M$  rather than  $N$ . We then scale the random variables by  $N / M$ .

```txt
beta = Normal(mu=tf.zeros([K, D]), sigma=tf.ones([K, D]))  
z = Categorical(logits=tf.zeros([M, K]))  
x = Normal(mu=tf.gather(beta, z), sigma=tf.ones([M, D]))  
qbeta = Normal(mu=tf.Variable(tf.zeros([K, D))), sigma=tf.nn.softplus(tf.Variable(tf.zeros[K, D]))  
qz = Categorical(logits=tf.Variable(tf.zeros[M, D]))  
inference = ed.VariationalInference({beta: qbeta, z: qz}, data={x: x_batch})  
inference.initialize(scale={x: float(N) / M, z: float(N) / M})
```

Conceptually, the scale argument represents scaling for each random variable's plate, as if we had seen that random variable  $N / M$  as many times. As an example, Appendix B shows how to implement stochastic variational inference in Edward. The approach extends naturally to streaming data (Doucet et al., 2000; Broderick et al., 2013; McInerney et al., 2015), dynamic batch sizes, and data structures in which working on a subgraph does not immediately apply (Binder et al., 1997; Johnson & Willsky, 2014; Foti et al., 2014).

# 5 EXPERIMENTS

In this section, we illustrate two main benefits of Edward. First, we show how it is easy to compare different inference algorithms on the same model. Second, we show how it is easy to get significant speedups by exploiting computational graphs.

<table><tr><td>Inference method</td><td>Negative log-likelihood</td></tr><tr><td>VAE (Kingma &amp; Welling, 2014)</td><td>≤ 88.2</td></tr><tr><td>VAE without analytic KL</td><td>≤ 89.4</td></tr><tr><td>VAE with analytic entropy</td><td>≤ 88.1</td></tr><tr><td>VAE with score function gradient</td><td>≤ 87.9</td></tr><tr><td>Normalizing flows (Rezende &amp; Mohamed, 2015)</td><td>≤ 85.8</td></tr><tr><td>Hierarchical variational model (Ranganath et al., 2016b)</td><td>≤ 85.4</td></tr><tr><td>Importance-weighted auto-encoders (K=50) (Burda et al., 2016)</td><td>≤ 86.3</td></tr><tr><td>HVM with IwAE objective (K=5)</td><td>≤ 85.2</td></tr><tr><td>Rényi divergence (α=-1) (Li &amp; Turner, 2016)</td><td>≤ 140.5</td></tr></table>

Table 1: Inference methods for a probabilistic decoder on binarized MNIST. The Edward PPL makes it easy to experiment with many algorithms.

# 5.1 RECENT METHODS IN VARIATIONAL INFERENCE

We demonstrate Edward's flexibility for experimenting with complex inference algorithms. We consider the VAE setup from Figure 2 and the binarized MNIST data set (Salakhutdinov & Murray, 2008). We use  $d = 50$  latent variables per data point and optimize using ADAM. We study different components of the VAE setup using different methods; Appendix C.1 is a complete script. After training we evaluate held-out log likelihoods, which are lower bounds on the true value.

Table 1 shows the results. The first method uses the VAE from Figure 2. The next three methods use the same VAE but apply different gradient estimators: reparameterization gradient without an analytic KL; reparameterization gradient with an analytic entropy; and the score function gradient (Paisley et al., 2012; Ranganath et al., 2014). This typically leads to the same optima but at different convergence rates. The score function gradient was slowest. Gradients with an analytic entropy produced difficulties around convergence: we switched to stochastic estimates of the entropy as it approached an optima. We also use hierarchical variational models (HVMs) (Ranganath et al., 2016b) with a normalizing flow prior; it produced similar results as a normalizing flow on the latent variable space (Rezende & Mohamed, 2015), and better than importance-weighted auto-encoders (IWAES) (Burda et al., 2016).

We also study several novel combinations, such as HVMs with the IwAE objective, GAN-based optimization on the decoder, and Renyi divergence on the decoder. GAN-based optimization does not enable calculation of the log-likelihood; Renyi divergence does not directly optimize for log-likelihood so it does not perform well. The key point is that these are easy modifications to scripts in Edward.

# 5.2 GPU-ACCELERATED HAMILTONIAN MONTE CARLO

![](images/9f7b7625efbe9a11d762ee64da7dc86b3ce9697761a1b821a5ff10fb3271362c.jpg)  
Figure 9: Edward program for Bayesian logistic regression with Hamiltonian Monte Carlo (HMC).

We analyze the efficiency to generate posterior samples with Hamiltonian Monte Carlo (HMC; Neal, 2011) on modern hardware—a desktop machine with a 12-core Intel i7-5930K CPU running at  $3.50\mathrm{GHz}$ , and an NVIDIA Titan X (Maxwell) GPU. We do posterior inference on a simple Bayesian logistic regression model using Edward, Stan (Carpenter et al., 2016), and PyMC3 (Salvatier et al., 2015). In Edward, this is implemented in Figure 9.

We ran four experiments on the Covertype dataset ( $N = 581012$ ,  $D = 54$ ; responses were binarized). In all experiments we ran 100 HMC iterations, with 10 leapfrog updates per iteration and a

step size of  $0.5 / N$ . Stan, running on one CPU core, took 171 seconds; PyMC3 took 361 seconds running on 12 CPU cores (PyMC3 was actually slower when using the GPU); Edward took 8.2 seconds running on 12 CPU cores; and Edward took 4.9 seconds running on the GPU. (These numbers exclude compilation time, which is significant for Stan.) The dramatic  $35\mathrm{x}$  speedup from Stan to Edward (GPU) showcases the value of building a PPL on top of computational graphs.

# 5.3 PROBABILITY ZOO

In addition to Edward, we also release the Probability Zoo, a community repository for pre-trained probability models and their posteriors. It was inspired by the model zoo in Caffe (Jia et al., 2014), which provides many pre-trained discriminative neural networks. It is also inspired by Forest (Stuhlmüller, 2012), which provides examples of probabilistic programs. Other examples in the Probability Zoo are discussed in the Appendix.

# 6 DISCUSSION

We have proposed Edward, a new Turing-complete PPL that provides compositional representations for probabilistic models and inference algorithms. This enables us to implement state-of-the-art techniques in probabilistic modeling, such as expressive variational inference and generative adversarial networks, as well as more traditional Bayesian hierarchical modeling. We also showed how we can leverage computational graphs to achieve fast computation and scale to massive data. Edward expands the scope of probabilistic programming to be as computationally efficient and flexible as traditional deep learning.

# ACKNOWLEDGEMENTS

We thank the Google BayesFlow team—Joshua Dillon, Ian Langmore, Ryan Sepassi, and Srinivas Vasudevan—as well as Amr Ahmed, Matthew Johnson, Hung Bui, Rajesh Ranganath, Maja Rudolph, and Francisco Ruiz for their helpful feedback and comments. This work is supported by NSF IIS-1247664, ONR N00014-11-1-0651, DARPA FA8750-14-2-0009, DARPA N66001-15-C-4032, Adobe, Google, NSERC PGS-D, and the Sloan Foundation.

# REFERENCES

Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zhang. TensorFlow: A system for large-scale machine learning. arXiv preprint arXiv:1605.08695, 2016.  
Christophe Andrieu and Gareth O Roberts. The pseudo-marginal approach for efficient Monte Carlo computations. The Annals of Statistics, pp. 697-725, 2009.  
John Binder, Kevin Murphy, and Stuart Russell. Space-efficient inference in dynamic probabilistic networks. In International Joint Conference on Artificial Intelligence, 1997.  
Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.  
David M Blei, Andrew Y Ng, and Michael I Jordan. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993-1022, 2003.  
Léon Bottou. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010, pp. 177-186. Springer, 2010.  
Tamara Broderick, Nicholas Boyd, Andre Wibisono, Ashia C Wilson, and Michael I Jordan. Streaming Variational Bayes. In Neural Information Processing Systems, pp. 1727-1735, 2013.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. In International Conference on Learning Representations, 2016.

Bob Carpenter, Andrew Gelman, Matthew D Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. Stan: A probabilistic programming language. Journal of Statistical Software, 2016.  
David E Culler. Dataflow architectures. Technical report, DTIC Document, 1986.  
Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a Laplacian pyramid of adversarial networks. In Neural Information Processing Systems, 2015.  
Adji B. Dieng, Dustin Tran, Rajesh Ranganath, John Paisley, and David M. Blei.  $\chi$ -divergence for approximate inference. In arXiv preprint arXiv:1611.00328, 2016.  
Arnaud Doucet, Simon Godsill, and Christophe Andrieu. On sequential Monte Carlo sampling methods for Bayesian filtering. Statistics and Computing, 10(3):197-208, 2000.  
Arnaud Doucet, Nando De Freitas, and Neil Gordon. An introduction to sequential monte carlo methods. In *Sequential Monte Carlo methods in practice*, pp. 3-14. Springer, 2001.  
Nicholas Foti, Jason Xu, Dillon Laird, and Emily Fox. Stochastic variational inference for hidden Markov models. In Neural Information Processing Systems, 2014.  
Alan E Gelfand and Adrian FM Smith. Sampling-based approaches to calculating marginal densities. Journal of the American statistical association, 85(410):398-409, 1990.  
Zoubin Ghahramani. Probabilistic machine learning and artificial intelligence. Nature, 521(7553): 452-459, 2015.  
Ian Goodfellow, Jean Pouget-Abadie, M Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Neural Information Processing Systems, 2014.  
Noah Goodman, Vikash Mansinghka, Daniel M Roy, Keith Bonawitz, and Joshua B Tenenbaum. Church: A language for generative models. In Uncertainty in Artificial Intelligence, 2012.  
Karol Gregor, Ivo Danihelka, Alex Graves, Danilo J Rezende, and Daan Wierstra. DRAW: A Recurrent Neural Network For Image Generation. In International Conference on Machine Learning, 2015.  
Sergio Guadarrama and Nathan Silberman. TensorFlow Slim, 2016.  
M Gutmann and A Hyvarinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. Artificial Intelligence and Statistics, 2010.  
Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation, 14(8):1771-1800, 2002.  
Matthew Hoffman and David M. Blei. Structured stochastic variational inference. In Artificial Intelligence and Statistics, 2015.  
Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic variational inference. The Journal of Machine Learning Research, 14(1):1303-1347, 2013.  
Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv preprint arXiv:1408.5093, 2014.  
Matthew Johnson and Alan S Willsky. Stochastic variational inference for Bayesian time series models. In International Conference on Machine Learning, 2014.  
M. I. Jordan, Z. Ghahramani, T. S. Jaakkola, and L. K. Saul. An introduction to variational methods for graphical models. Machine Learning, 37(2):183-233, 1999.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations, 2014.  
Diederik P Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Neural Information Processing Systems, 2014.

Diederik P Kingma, Tim Salimans, and Max Welling. Improving Variational Inference with Inverse Autoregressive Flow. In Neural Information Processing Systems, 2016.  
Oleg Kiselyov and Chung-Chieh Shan. Embedded probabilistic programming. In Domain-Specific Languages, pp. 360-384. Springer, 2009.  
Daphne Koller and Nir Friedman. Probabilistic graphical models: principles and techniques. MIT press, 2009.  
Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building Machines That Learn and Think Like People. arXiv preprint arXiv:1604.00289, 2016.  
Yingzhen Li and Richard E Turner. Variational inference with Rényi divergence. In Neural Information Processing Systems, 2016.  
Yujia Li, Kevin Swersky, and Richard Zemel. Generative moment matching networks. In International Conference on Machine Learning, 2015.  
V Mansinghka, D Selsam, and Y Perov. Venture: A higher-order probabilistic programming platform with programmable inference. arXiv.org, 2014.  
James McInerney, Rajesh Ranganath, and David M Blei. The Population Posterior and Bayesian Inference on Streams. In Neural Information Processing Systems, 2015.  
Brian Milch, Bhaskara Marthi, Stuart Russell, David Sontag, Daniel L Ong, and Andrey Kolobov. Blog. In International Joint Conference on Artificial Intelligence, 2005.  
T. Minka, J.M. Winn, J.P. Guiver, S. Webster, Y. Zaykov, B. Yangel, A. Spengler, and J. Bronskill. Infer.NET 2.6, 2014. Microsoft Research Cambridge. http://research.microsoft.com/infernet.  
Thomas P Minka. Expectation propagation for approximate Bayesian inference. In Uncertainty in Artificial Intelligence, 2001.  
Kevin Murphy. The Bayes net toolbox for Matlab. Computing Science and Statistics, 33(2):1024-1034, 2001.  
Radford M Neal. MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, 2011.  
Radford M. Neal and Geoffrey E. Hinton. A new view of the em algorithm that justifies incremental and other variants. In Learning in Graphical Models, pp. 355-368. Kluwer Academic Publishers, 1993.  
John Paisley, David M. Blei, and Michael Jordan. Variational Bayesian inference with stochastic search. In International Conference on Machine Learning, 2012.  
Avi Pfeffer. IBAL: A probabilistic rational programming language. In International Joint Conference on Artificial Intelligence, pp. 733-740. CiteSeer, 2001.  
Avi Pfeffer. Figaro: An object-oriented probabilistic programming language. Charles River Analytics Technical Report, 137, 2009.  
Martyn Plummer. JAGS: a program for analysis of Bayesian graphical models using Gibbs sampling. In International Workshop on Distributed Statistical Computing, 2003.  
Rajesh Ranganath, Sean Gerrish, and David M. Blei. Black box variational inference. In Artificial Intelligence and Statistics, 2014.  
Rajesh Ranganath, Jaan Altosaar, Dustin Tran, and David M. Blei. Operator variational inference. In Neural Information Processing Systems, 2016a.  
Rajesh Ranganath, Dustin Tran, and David M Blei. Hierarchical variational models. In International Conference on Machine Learning, 2016b.  
Danilo J Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, 2015.

Danilo J Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, 2014.  
Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, and Daan Wierstra. One-shot generalization in deep generative models. In International Conference on Machine Learning, 2016.  
Daniel Ritchie, Paul Horsfall, and Noah D Goodman. Deep amortized inference for probabilistic programs. arXiv preprint arXiv:1610.05735, 2016.  
Christian P Robert and George Casella. Monte Carlo Statistical Methods. Springer, 1999.  
Maja R Rudolph, Francisco J R Ruiz, Stephan Mandt, and David M Blei. Exponential family embeddings. In Neural Information Processing Systems, 2016.  
Ruslan Salakhutdinov and Geoffrey E Hinton. Deep Boltzmann machines. In Artificial Intelligence and Statistics, 2009.  
Ruslan Salakhutdinov and Iain Murray. On the quantitative analysis of deep belief networks. In International Conference on Machine Learning, 2008.  
John Salvatier, Thomas Wiecki, and Christopher Fonnesbeck. Probabilistic Programming in Python using PyMC. arXiv preprint arXiv:1507.08050, 2015.  
Adam Scibior, Zoubin Ghahramani, and Andrew D Gordon. Practical probabilistic programming with monads. In the 8th ACM SIGPLAN Symposium, pp. 165-176, New York, New York, USA, 2015. ACM Press.  
David J Spiegelhalter, Andrew Thomas, Nicky G Best, and Wally R Gilks. BUGS: Bayesian inference using Gibbs sampling, version 0.50. MRC Biostatistics Unit, Cambridge, 1995.  
Andreas Stuhlmüller. Forest: A repository for generative models, 2012. URL http:// forestdb.org.  
Dustin Tran, Alp Kucukelbir, Adji B. Dieng, Maja Rudolph, Dawen Liang, and David M. Blei. Edward: A library for probabilistic modeling, inference, and criticism. arXiv preprint arXiv:1610.09787, 2016a.  
Dustin Tran, Rajesh Ranganath, and David M. Blei. The variational Gaussian process. In International Conference on Learning Representations, 2016b.  
Jean-Baptiste Tristan, Daniel Huang, Joseph Tassarotti, Adam C Pocock, Stephen Green, and Guy L Steele. Augur: Data-parallel probabilistic modeling. In Neural Information Processing Systems, 2014.  
Chong Wang and David M Blei. Truncation-free online variational inference for Bayesian nonparametric models. In Neural Information Processing Systems, pp. 413-421, 2012.  
Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient Langevin dynamics. In International Conference on Machine Learning, 2011.  
Frank Wood, Jan Willem van de Meent, and Vikash Mansinghka. A new approach to probabilistic programming inference. In Artificial Intelligence and Statistics, 2014.  
Yi Wu, Lei Li, Stuart Russell, and Rastislav Bodik. Swift: Compiled inference for probabilistic programming languages. arXiv preprint arXiv:1606.09242, 2016.  
Robert Zinkov and Chung-chieh Shan. Composing inference algorithms as program transformations. arXiv preprint arXiv:1603.01882, 2016.
