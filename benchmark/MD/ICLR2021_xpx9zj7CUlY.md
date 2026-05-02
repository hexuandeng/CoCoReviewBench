# RANDOMIZED AUTOMATIC DIFFERENTIATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

The successes of deep learning, variational inference, and many other fields have been aided by specialized implementations of reverse-mode automatic differentiation (AD) to compute gradients of mega-dimensional objectives. The AD techniques underlying these tools were designed to compute exact gradients to numerical precision, but modern machine learning models are almost always trained with stochastic gradient descent. Why spend computation and memory on exact (minibatch) gradients only to use them for stochastic optimization? We develop a general framework and approach for randomized automatic differentiation (RAD), which can allow unbiased gradient estimates to be computed with reduced memory in return for variance. We examine limitations of the general approach, and argue that we must leverage problem specific structure to realize benefits. We develop RAD techniques for a variety of simple neural network architectures, and show that for a fixed memory budget, RAD converges in fewer iterations than using a small batch size for feedforward networks, and in a similar number for recurrent networks. We also show that RAD can be applied to scientific computing, and use it to develop a low-memory stochastic gradient method for optimizing the control parameters of a linear reaction-diffusion PDE representing a fission reactor.

# 1 INTRODUCTION

Deep neural networks have taken center stage as a powerful way to construct and train massively-parametric machine learning (ML) models for supervised, unsupervised, and reinforcement learning tasks. There are many reasons for the resurgence of neural networks—large data sets, GPU numerical computing, technical insights into overparameterization, and more—but one major factor has been the development of tools for automatic differentiation (AD) of deep architectures. Tools like PyTorch and TensorFlow provide a computational substrate for rapidly exploring a wide variety of differentiable architectures without performing tedious and error-prone gradient derivations. The flexibility of these tools has enabled a revolution in AI research, but the underlying ideas for reverse-mode AD go back decades. While tools like PyTorch and TensorFlow have received huge dividends from a half-century of AD research, they are also burdened by the baggage of design decisions made in a different computational landscape. The research on AD that led to these ubiquitous deep learning frameworks is focused on the computation of Jacobians that are exact up to numerical precision. However, in modern workflows these Jacobians are used for stochastic optimization. We ask:

Why spend resources on exact gradients when we're going to use stochastic optimization?

This question is motivated by the surprising realization over the past decade that deep neural network training can be performed almost entirely with first-order stochastic optimization. In fact, empirical evidence supports the hypothesis that the regularizing effect of gradient noise assists model generalization (Keskar et al., 2017; Smith & Le, 2018; Hochreiter & Schmidhuber, 1997). Stochastic gradient descent variants such as AdaGrad (Duchi et al., 2011) and Adam (Kingma & Ba, 2015) form the core of almost all successful optimization techniques for these models, using small subsets of the data to form the noisy gradient estimates.

The goals and assumptions of automatic differentiation as performed in classical and modern systems are mismatched with those required by stochastic optimization. Traditional AD computes the derivative or Jacobian of a function accurately to numerical precision. This accuracy is required for many problems in applied mathematics which AD has served, e.g., solving systems of differential equations. But in stochastic optimization we can make do with inaccurate gradients, as long as

from math import sin, exp

def f(x1, x2):

$\mathrm{a} = \exp (\mathrm{x}1)$

$\mathbf{b} = \sin (\mathbf{x}2)$

c = b * x2

$\mathrm{d} = \mathrm{a}*\mathrm{c}$

return a \* d

(a) Differentiable Python function

![](images/88ce443262f3ff115bd3a0391e872238d7bc2683123a275715eaeec01c4d1081.jpg)  
(b) Primal graph

![](images/4f3871a22d1c6f185d40f2e69de0fb5659157995de7a19de838a208c4000db05.jpg)  
(c) Linearized graph  
Figure 1: Illustration of the basic concepts of the linearized computational graph and Bauer's formula. (a) a simple Python function with intermediate variables; (b) the primal computational graph, a DAG with variables as vertices and flow moving upwards to the output; (c) the linearized computational graph (LCG) in which the edges are labeled with the values of the local derivatives; (d) illustration of the four paths that must be evaluated to compute the Jacobian. (Example from Paul D. Hovland.)

![](images/a89231134a64b652c32d9d3a253ce0737288389291ce4eeec0751f973eaf4066.jpg)  
(d) Bauer paths

our estimator is unbiased and has reasonable variance. We ask the same question that motivates mini-batch SGD: why compute an exact gradient if we can get noisy estimates cheaply? By thinking of this question in the context of AD, we can go beyond mini-batch SGD to more general schemes for developing cheap gradient estimators: in this paper, we focus on developing gradient estimators with low memory cost. Although previous research has investigated approximations in the forward or reverse pass of neural networks to reduce computational requirements, here we replace deterministic AD with randomized automatic differentiation (RAD), trading off of computation for variance inside AD routines when imprecise gradient estimates are tolerable, while retaining unbiasedness.

# 2 AUTOMATIC DIFFERENTIATION

Automatic (or algorithmic) differentiation is a family of techniques for taking a program that computes a differentiable function  $f: \mathbb{R}^n \to \mathbb{R}^m$ , and producing another program that computes the associated derivatives; most often the Jacobian:  $\mathcal{J}[f] = f': \mathbb{R}^n \to \mathbb{R}^{m \times n}$ . (For a comprehensive treatment of AD, see Griewank & Walther (2008); for an ML-focused review see Baydin et al. (2018).) In most machine learning applications,  $f$  is a loss function that produces a scalar output, i.e.,  $m = 1$ , for which the gradient with respect to parameters is desired. AD techniques are contrasted with the method of finite differences, which approximates derivatives numerically using a small but non-zero step size, and also distinguished from symbolic differentiation in which a mathematical expression is processed using standard rules to produce another mathematical expression, although Elliott (2018) argues that the distinction is simply whether or not it is the compiler that manipulates the symbols.

There are a variety of approaches to AD: source-code transformation (e.g., Bischof et al. (1992); Hascoet & Pascual (2013); van Merrienboer et al. (2018)), execution tracing (e.g., Walther & Griewank (2009); Maclaurin et al.), manipulation of explicit computational graphs (e.g., Abadi et al. (2016); Bergstra et al. (2010)), and category-theoretic transformations (Elliott, 2018). AD implementations exist for many different host languages, although they vary in the extent to which they take advantage of native programming patterns, control flow, and language features. Regardless of whether it is constructed at compile-time, run-time, or via an embedded domain-specific language, all AD approaches can be understood as manipulating the linearized computational graph (LCG) to collapse out intermediate variables. Figure 1 shows the LCG for a simple example. These computational graphs are always directed acyclic graphs (DAGs) with vertices as variables.

Let the outputs of  $f$  be  $y_{j}$ , the inputs  $\theta_{i}$ , and the intermediates  $z_{l}$ . AD can be framed as the computation of a partial derivative as a sum over all paths through the LCG DAG (Bauer, 1974):

$$
\frac {\partial y _ {j}}{\partial \theta_ {i}} = \mathcal {J} _ {\theta} [ f ] _ {j, i} = \sum_ {[ i \rightarrow j ] (k, l) \in [ i \rightarrow j ]} \frac {\partial z _ {l}}{\partial z _ {k}} \tag {1}
$$

where  $[i\to j]$  indexes paths from vertex  $i$  to vertex  $j$  and  $(k,l)\in [i\rightarrow j]$  denotes the set of edges in that path. See Figure 1d for an illustration. Although general, this naive sum over paths does not take advantage of the structure of the problem and so, as in other kinds of graph computations, dynamic programming (DP) provides a better approach. DP collapses substructures of the graph until

it becomes bipartite and the remaining edges from inputs to outputs represent exactly the entries of the Jacobian matrix. This is referred to as the Jacobian accumulation problem (Naumann, 2004) and there are a variety of ways to manipulate the graph, including vertex, edge, and face elimination (Griewank & Naumann, 2002). Forward-mode AD and reverse-mode AD (backpropagation) are special cases of more general dynamic programming strategies to perform this summation; determination of the optimal accumulation schedule is unfortunately NP-complete (Naumann, 2008).

While the above formulation in which each variable is a scalar can represent any computational graph, it can lead to structures that are difficult to reason about. Often we prefer to manipulate vectors and matrices, and we can instead let each intermediate  $z_{l}$  represent a  $d_{l}$  dimensional vector. In this case,  $\partial z_{l} / \partial z_{k}\in \mathbb{R}^{d_{l}\times d_{k}}$  represents the intermediate Jacobian of the operation  $z_{k}\rightarrow z_{l}$ . Note that eqn equation 1 now expresses the Jacobian of  $f$  as a sum over chained matrix products.

# 3 RANDOMIZING AUTOMATIC DIFFERENTIATION

We introduce techniques that could be used to decrease the resource requirements of AD when used for stochastic optimization. We focus on functions with a scalar output where we are interested in the gradient of the output with respect to some parameters,  $\mathcal{I}_{\theta}[f]$ . Reverse-mode AD efficiently calculates  $\mathcal{I}_{\theta}[f]$ , but requires the full linearized computational graph to either be stored during the forward pass, or to be recomputed during the backward pass using intermediate variables recorded during the forward pass. For large computational graphs this could provide a large memory burden.

The most common technique for reducing the memory requirements of AD is gradient checkpointing (Griewank & Walther, 2000; Chen et al., 2016), which saves memory by adding extra forward pass computations. Checkpointing is effective when the number of "layers" in a computation graph is much larger than the memory required at each layer. We take a different approach; we instead aim to save memory by increasing gradient variance, without extra forward computation.

Our main idea is to consider an unbiased estimator  $\hat{\mathcal{J}}_{\theta}[f]$  such that  $\mathbb{E}\hat{\mathcal{J}}_{\theta}[f] = \mathcal{J}_{\theta}[f]$  which allows us to save memory required for reverse-mode AD. Our approach is to determine a sparse (but random) linearized computational graph during the forward pass such that reverse-mode AD applied on the sparse graph yields an unbiased estimate of the true gradient. Note that the original computational graph is used for the forward pass, and randomization is used to determine a LCG to use for the backward pass in place of the original computation graph. We may then decrease memory costs by storing the sparse LCG directly or storing intermediate variables required to compute the sparse LCG.

In this section we provide general recipes for randomizing AD by sparsifying the LCG. In sections 4 and 5 we apply these recipes to develop specific algorithms for neural networks and linear PDEs which achieve concrete memory savings.

# 3.1 PATH SAMPLING

Observe that in Bauer's formula each Jacobian entry is expressed as a sum over paths in the LCG. A simple strategy is to sample paths uniformly at random from the computation graph, and form a Monte Carlo estimate of Equation 1. Naively this could take multiple passes through the graph. However, multiple paths can be sampled without significant computation overhead by performing a topological sort of the vertices and iterating through vertices, sampling multiple outgoing edges for each. We provide a proof and detailed algorithm in the appendix. Dynamic programming methods such as reverse-mode automatic differentiation can then be applied to the sparsified LCG.

# 3.2 RANDOM MATRIX INJECTION

In computation graphs consisting of vector operations, the vectorized computation graph is a more compact representation. We introduce an alternative view on sampling paths in this case. A single path in the vectorized computation graph represents many paths in the underlying scalar computation graph. As an example, Figure 2c is a vector representation for Figure 2b. For this example,

$$
\frac {\partial y}{\partial \theta} = \frac {\partial y}{\partial C} \frac {\partial C}{\partial B} \frac {\partial B}{\partial A} \frac {\partial A}{\partial \theta} \tag {2}
$$

where  $A, B, C$  are vectors with entries  $a_i, b_i, c_i, \frac{\partial C}{\partial B}, \frac{\partial B}{\partial A}$  are  $3 \times 3$  Jacobian matrices for the intermediate operations,  $\frac{\partial y}{\partial C}$  is  $1 \times 3$ , and  $\frac{\partial A}{\partial \theta}$  is  $3 \times 1$ .

We now note that the contribution of the path  $p = \theta \rightarrow a_1 \rightarrow b_2 \rightarrow c_2 \rightarrow y$  to the gradient is,

$$
\frac {\partial y}{\partial C} P _ {2} \frac {\partial C}{\partial B} P _ {2} \frac {\partial B}{\partial A} P _ {1} \frac {\partial A}{\partial \theta} \tag {3}
$$

where  $P_{i} = e_{i}e_{i}^{T}$  (outer product of standard basis vectors). Sampling from  $\{P_1,P_2,P_3\}$  and right multiplying a Jacobian is equivalent to sampling the paths passing through a vertex in the scalar graph.

In general, if we have transition  $B \to C$  in a vectorized computational graph, where  $B \in \mathbb{R}^d$ ,  $C \in \mathbb{R}^m$ , we can insert a random matrix  $P = {}^{d}/k \sum_{s=1}^{k} P_s$  where each  $P_s$  is sampled uniformly from  $\{P_1, P_2, \ldots, P_d\}$ . With this construction,  $\mathbb{E}P = I_d$ , so

$$
\mathbb {E} \left[ \frac {\partial C}{\partial B} P \right] = \frac {\partial C}{\partial B}. \tag {4}
$$

Right multiplication by  $P$  may be achieved by sampling the intermediate Jacobian: one does not need to actually assemble and multiply the two matrices. For clarity we adopt the notation  $S_P[\partial C / \partial B] = \partial C / \partial BP$ . This is sampling (with replacement)  $k$  out of the  $d$  vertices represented by  $B$ , and only considering paths that pass from those vertices.

The important properties of  $P$  that enable memory savings with an unbiased approximation are

$$
\mathbb {E} P = I _ {d} \quad \text {a n d} \quad P = R R ^ {T}, R \in \mathbb {R} ^ {d \times k}, k <   d. \tag {5}
$$

We could therefore consider other matrices with the same properties. In our additional experiments in the appendix, we also let  $R$  be a random projection matrix of independent Rademacher random variables compressed sensing and randomized dimensionality reduction.

In vectorized computational graphs, we can imagine a two-level sampling scheme. We can both sample paths from the computational graph where each vertex on the path corresponds to a vector. We can also sample within each vector path, with sampling performed via matrix injection as above.

In many situations the full intermediate Jacobian for a vector operation is unreasonable to store. Consider the operation  $B \to C$  where  $B, C \in \mathbb{R}^d$ . The Jacobian is  $d \times d$ . Thankfully many common operations are element-wise, leading to a diagonal Jacobian that can be stored as a  $d$ -vector. Another common operation is matrix-vector products. Consider  $Ab = c$ ,  $\frac{\partial c}{\partial b} = A$ . Although  $A$  has many more entries than  $c$  or  $b$ , in many applications  $A$  is either a parameter to be optimized or is easily recomputed. Therefore in our implementations, we do not directly construct and sparsify the Jacobians. We instead sparsify the input vectors or the compact version of the Jacobian in a way that has the same effect. Unfortunately, there are some practical operations such as softmax that do not have a compactly-representable Jacobian and for which this is not possible.

# 3.3 VARIANCE

The variance incurred by path sampling and random matrix injection will depend on the structure of the LCG. We present two extremes in Figure 2. In Figure 2a, each path is independent and there are a small number of paths. If we sample a fixed fraction of all paths, variance will be constant in the depth of the graph. In contrast, in Figure 2b, the paths overlap, and the number of paths increases exponentially with depth. Sampling a fraction of outgoing edges for each vertex will lead to an exponentially decreasing fraction of paths sampled, and exponentially increasing variance with depth.

It is thus difficult to apply sampling schemes without knowledge of the underlying graph. Indeed, our initial efforts to apply random matrix injection schemes to neural network graphs resulted in variance exponential with depth of the network, which prevented stochastic optimization from converging. We develop tailored sampling strategies for computation graphs corresponding to problems of common interest, exploiting properties of these graphs to avoid the exploding variance problem.

![](images/8bf9d9bda9e731f9d037c41e6fb8bbb7b31f19a4f3cd15e0dcc818e98632b2ed.jpg)

![](images/2876cc3989e861a46b1a9e46323da798ed036f2df559270ba75ae000fc3c2f4a.jpg)  
(a) Independent Paths Graph

![](images/307427789b04b6b96dd3568c5005a4e7ecb697a4a9cba531e6b75e6bbe9da132.jpg)  
(b) Fully Interleaved Graph  
(c) Vector graph for (b).  
Figure 2: Common computational graph patterns. The graphs may be arbitrarily deep and wide. (a) A small number of independent paths. Path sampling has constant variance with depth. (b) The number of paths increases exponentially with depth; path sampling gives high variance. Independent paths are common when a loss decomposes over data. Fully interleaved graphs are common with vector operations.

# 4 CASE STUDY: NEURAL NETWORKS

We consider neural networks composed of fully connected layers, convolution layers, ReLU nonlinearities, and pooling layers. We take advantage of the important property that many of the intermediate Jacobians can be compactly stored, and the memory required during reverse-mode is often bottlenecked by a few operations. We draw a vectorized computational graph for a typical simple neural network in figure 3. Although the diagram depicts a dataset of size of 3, batch size of size 1, and 2 hidden layers, we assume the dataset size is  $N$ . Our analysis is valid for any number of hidden layers, and also recurrent networks. We are interested in the gradients  $\frac{\partial y}{\partial W_1}$  and  $\frac{\partial y}{\partial W_2}$ .

# 4.1 MINIBATCH SGD AS RANDOMIZED AD

At first look, the diagram has a very similar pattern to that of 2a, so that path sampling would be a good fit. Indeed, we could sample  $\bar{B} < N$  paths from  $W_{1}$  to  $y$ , and also  $B$  paths from  $W_{2}$  to  $y$ . Each path corresponds to processing a different batch element, and the computations are independent.

In empirical risk minimization, the final loss function is an average of the loss over data points. Therefore, the intermediate partials  $\partial y / \partial h_{2,x}$  for each data point  $x$  will be independent of the other data points. As a result, if the same paths are chosen in path sampling for  $W_{1}$  and  $W_{2}$ , and if we are only interested in the stochastic gradient (and not the full function evaluation), the computation graph only needs to be evaluated for the data points corresponding to the sampled paths. This exactly corresponds to mini-batching. The paths are visually depicted in Figure 3b.

# 4.2 ALTERNATIVE SGD SCHEMES WITH RANDOMIZED AD

We wish to use our principles to derive a randomization scheme that can be used on top of mini-batch SGD. We ensure our estimator is unbiased as we randomize by applying random matrix injection independently to various intermediate Jacobians. As the expectation of a product of independent random variables is equal to the product of their expectations, the resulting estimator is unbiased. Consider a path corresponding to data point 1. The contribution to the gradient  $\partial y / \partial W_1$  is

$$
\frac {\partial y}{\partial h _ {2 , 1}} \frac {\partial h _ {2 , 1}}{\partial a _ {1 , 1}} \frac {\partial a _ {1 , 1}}{\partial h _ {1 , 1}} \frac {\partial h _ {1 , 1}}{\partial W _ {1}} \tag {6}
$$

Using random matrix injection to sample every Jacobian would lead to exploding variance. Instead, we analyze each term to see which are memory bottlenecks.

$\partial y / \partial h_{2,1}$  is the Jacobian with respect to (typically) the loss. Memory requirements for this Jacobian are independent of depth of the network. The dimension of the classifier

![](images/e0e54467f635f1a8a9e64710fb88c3a0435bedce712b79ba4451215333b9e646.jpg)

![](images/9adaa0b06da78b644e3c514db431959fd13a050d8b4d1ffd535108f66221be07.jpg)  
(a) Neural network computational graph  
(b) Computational Graph with Mini-batching  
Figure 3: NN computation graphs.

is usually smaller (10 - 1000) than the other layers (which can have dimension 10,000 or more in convolutional networks). Therefore, the Jacobian at the output layer is not a memory bottleneck.

$\partial h_{2,1} / \partial a_{1,1}$  is the Jacobian of the hidden layer with respect to the previous layer activation. This can be constructed from  $W_{2}$ , which must be stored in memory, with memory cost independent of batch size. In convnets, due to weight sharing, the effective dimensionality is much smaller than  $H_{1} \times H_{2}$ . In recurrent networks, it is shared across timesteps. Therefore, these are not a memory bottleneck.

$\partial a_{1,1} / \partial h_{1,1}$  contains the Jacobian of the ReLU activation function. This can be compactly stored using 1-bit per entry, as the gradient can only be 1 or 0. Note that this is true for ReLU activations in particular, and not true for general activation functions, although ReLU is widely used in deep learning. For ReLU activations, these partials are not a memory bottleneck.

$\partial h_{1,1} / \partial W_1$  contains the memory bottleneck for typical ReLU neural networks. This is the Jacobian of the hidden layer output with respect to  $W_{1}$ , which, in a multi-layer perceptron, is equal to  $x_{1}$ . For  $B$  data points, this is a  $B\times D$  dimensional matrix.

![](images/ea192cdaefbd665d27f54cde4325da1a912d1c765c66b24619bb185a44eddf13.jpg)  
Figure 4: Convnet activation sampling for one batch element.  $X$  is the image,  $H$  is the pre-activation, and  $A$  is the activation.  $A$  is the output of a ReLU, so we can store the Jacobian  $\partial A_1 / \partial H_1$  with 1 bit per entry. For  $X$  and  $H$  we sample spatial elements and compute the Jacobians  $\partial H_1 / \partial W_1$  and  $\partial H_2 / \partial W_2$  with the sparse tensors.

Accordingly, we choose to sample  $\frac{\partial h_{1,1}}{\partial W_1}$ , replacing the matrix chain with  $\frac{\partial y}{\partial h_{2,1}}\frac{\partial h_{2,1}}{\partial a_{1,1}}\frac{\partial a_{1,1}}{\partial h_{1,1}}\mathcal{S}_{P_{W_1}}\left[\frac{\partial h_{1,1}}{\partial W_1}\right]$ . For an arbitrarily deep NN, this can be generalized:

$$
\frac {\partial y}{\partial h _ {d , 1}} \frac {\partial h _ {d , 1}}{\partial a _ {d - 1 , 1}} \frac {\partial a _ {d - 1 , 1}}{\partial h _ {d - 1 , 1}} \dots \frac {\partial a _ {1 , 1}}{\partial h _ {1 , 1}} \mathcal {S} _ {P W _ {1}} \left[ \frac {\partial h _ {1 , 1}}{\partial W _ {1}} \right], \quad \frac {\partial y}{\partial h _ {d , 1}} \frac {\partial h _ {d , 1}}{\partial a _ {d - 1 , 1}} \frac {\partial a _ {d - 1 , 1}}{\partial h _ {d - 1 , 1}} \dots \frac {\partial a _ {2 , 1}}{\partial h _ {2 , 1}} \mathcal {S} _ {P W _ {2}} \left[ \frac {\partial h _ {2 , 1}}{\partial W _ {2}} \right]
$$

This can be interpreted as sampling activations on the backward pass. This is our proposed alternative SGD scheme for neural networks: along with sampling data points, we can also sample activations, while maintaining an unbiased approximation to the gradient. This does not lead to exploding variance, as along any path from a given neural network parameter to the loss, the sampling operation is only applied to a single Jacobian. Sampling for convolutional networks is visualized in Figure 4.

# 4.3 NEURAL NETWORK EXPERIMENTS

We evaluate our proposed RAD method on two feedforward architectures: a small fully connected network trained on MNIST, and a small convolutional network trained on CIFAR-10. We also evaluate our method on an RNN trained on Sequential-MNIST. The exact architectures and the calculations for the associated memory savings from our method are available in the appendix. In the appendix we also include empirical analysis of gradient noise caused by RAD vs mini-batching.

We are mainly interested in the following question:

For a fixed memory budget and fixed number of gradient descent iterations, how quickly does our proposed method optimize the training loss compared to standard SGD with a smaller batch size?

Reducing the batch size will also reduce computational costs, while RAD will only reduce memory costs. Theoretically our method could reduce computational costs slightly, but this is not our focus. We only consider the memory/gradient variance tradeoff while avoiding adding significant overhead on top of vanilla reverse-mode (as is the case for checkpointing).

Results are shown in figure 5. Our feedforward network full-memory baseline is trained with a batch size of 150. For RAD we keep a batch size of 150, and try 2 different configurations. For "same sample", we sample with replacement a 0.1 fraction of activations, and the same activations are sampled for each batch element. For "different sample", we sample a 0.1 fraction of activations, independently for each batch element. Our "reduced batch" experiment is trained without RAD with a batch size of 20 for CIFAR-10 and 22 for MNIST. This achieves similar memory budget as RAD with batch size 150. Details of this calculation and of hyperparameters are in the appendix.

For the feedforward networks we tune the learning rate and  $\ell_2$  regularization parameter separately for each gradient estimator on a randomly held out validation set. We train with the best performing hyperparameters on bootstrapped versions of the full training set to measure variability in training. Details are in the appendix, including plots for train/test accuracy/loss, and a wider range of fraction of activations sampled. All feedforward models are trained with Adam.

In the RNN case, we also run baseline, "same sample", "different sample" and "reduced batch" experiments. The "reduced batch" experiment used a batch size of 21, while the others used a batch

<table><tr><td>Fraction of activations</td><td>Baseline (1.0)</td><td>0.8</td><td>0.5</td><td>0.3</td><td>0.1</td><td>0.05</td></tr><tr><td>ConvNet Mem</td><td>23.08</td><td>19.19</td><td>12.37</td><td>7.82</td><td>3.28</td><td>2.14</td></tr><tr><td>Fully Connected Mem</td><td>2.69</td><td>2.51</td><td>2.21</td><td>2.00</td><td>1.80</td><td>1.75</td></tr><tr><td>RNN Mem</td><td>47.93</td><td>39.98</td><td>25.85</td><td>16.43</td><td>7.01</td><td>4.66</td></tr></table>

Table 1: Peak memory (including weights) in MBytes used during training with 150 batch elements. Reported values are hand calculated and represent the expected memory usage of RAD under an efficient implementation.

![](images/76d89e1b516e54a7da495a075f93707766557afe706f28c26626258ca180a87d.jpg)  
(a) CIFAR-10 Curves

![](images/cfe08e4a6bcef5f2c6a46594e9c0b58aaefc890bd242988eded924f46f80c286.jpg)  
(b) MNIST Curves  
Figure 5: Training curves for neural networks. The legend in (c) applies to all plots. For the convolutional and fully connected neural networks, the loss decreases faster using activation sampling, compared to reducing the batch size further to match the memory usage. For the fully connected NN on MNIST, it is important to sample different activations for each batch element, since otherwise only part of the weight vector will get updated with each iteration. For the convolutional NN on CIFAR-10, this is not an issue due to weight tying. As expected, the full memory baseline converges quicker than the low memory versions. For the RNN on Sequential-MNIST, sampling different activations at each time-step matches the performance obtained by reducing the batch size.

![](images/9c0765c2e36c4a93b6f2c90aefa92aaeb62bbbc39c99665d0cf4d52038f02df3.jpg)  
(c) Sequential-MNIST Curves

size of 150. The learning rate was fixed at  $10^{-4}$  for all gradient estimators, found via a coarse grid search for the largest learning rate for which optimization did not diverge. Although we did not tune the learning rate separately for each estimator, we still expect that with a fixed learning rate, the lower variance estimators should perform better. When sampling, we sample different activations at each time-step. All recurrent models are trained with SGD without momentum.

# 5 CASE STUDY: REACTION-DIFFUSION PDE-CONSTRAINED OPTIMIZATION

Our second application is motivated by the observation that many scientific computing problems involve a repeated or iterative computation resulting in a layered computational graph. We may apply RAD to get a stochastic estimate of the gradient by subsampling paths through the computational graph. For certain problems, we can leverage problem structure to develop a low-memory stochastic gradient estimator without exploding variance. To illustrate this possibility we consider the optimization of a linear reaction-diffusion PDE on a square domain with Dirichlet boundary conditions, representing the production and diffusion of neutrons in a fission reactor (McClarren, 2018). Simulating this process involves solving for a potential  $\phi(x,y,t)$  varying in two spatial coordinates and in time. The solution obeys the partial differential equation:

$$
\frac {\partial \phi (x , y , t)}{\partial t} = D \nabla^ {2} \phi (x, y, t) + C (x, y, t, \boldsymbol {\theta}) \phi (x, y, t)
$$

We solve this PDE on a spatial grid using an explicit update rule  $\phi_{t + 1} = M\phi_t + \Delta tC_t\odot \phi_t$  The initial condition is  $\phi_0 = \sin (\pi x)\sin (\pi y)$  , with  $\phi = 0$  on the boundary of the domain. The loss function is the timeaveraged squared error between  $\phi$  and a time-dependent target,  $L = 1 / T\sum_{t}||\phi_{t}(\pmb {\theta}) - \phi_{t}^{\mathrm{target}}||_{2}^{2}$  The target is  $\phi_t^{\mathrm{target}} = \phi_0 + 1 / 4\sin (\pi t)\sin (2\pi x)\sin (\pi y)$  The source  $C$  is given by a seven-term Fourier series in  $x$  and  $t$  , with coefficients given by  $\pmb {\theta}\in \mathbb{R}^7$  , where  $\pmb{\theta}$  is the simulation details are provided in the appendix.

![](images/822b64695a19dd54e7e2db2bcd0f7f0a90c401172d53c87fa4c13a4ff8604351.jpg)

![](images/4c2bc1aa0c238149f53a3601ac7c608692ca120c3f77132c2cc6b6761ec44b67.jpg)  
(a) Visualization of sampling  
(b) RAD curves  
Figure 6: Reaction-diffusion PDE expt. (b) RAD saves up to  $99\%$  of memory without significant slowdown in convergence.

The gradient is  $\frac{\partial L}{\partial\theta} = \sum_{t = 1}^{T}\frac{\partial L}{\partial\phi_t}\sum_{i = 1}^t\left(\prod_{j = i}^{t - 1}\frac{\partial\phi_{j + 1}}{\partial\phi_j}\right)\frac{\partial\phi_i}{\partial C_{i - 1}}\frac{\partial C_{i - 1}}{\partial\theta}$ . As the reaction-diffusion PDE is linear and explicit,  $\frac{\partial\phi_{j + 1}}{\partial\phi_j}\in \mathbb{R}^{N_x^2\times N_x^2}$  is known and independent of  $\phi$ . We avoid storing  $C$  at each timestep by recomputing  $C$  from  $\pmb{\theta}$  and  $t$ . This permits a low-memory stochastic gradient estimate without exploding variance by sampling from  $\frac{\partial L}{\partial\phi_t}\in \mathbb{R}^{N_x^2}$  and the diagonal matrix  $\frac{\partial\phi_i}{\partial C_{i - 1}}$

replacing  $\frac{\partial L}{\partial\theta}$  with the unbiased estimator

$$
\sum_ {t = 1} ^ {T} \mathcal {S} _ {P \phi_ {t}} \left[ \frac {\partial L}{\partial \phi_ {t}} \right] \sum_ {i = 1} ^ {t} \left(\prod_ {j = i} ^ {t - 1} \frac {\partial \phi_ {j + 1}}{\partial \phi_ {j}}\right) \mathcal {S} _ {P \phi_ {i - 1}} \left[ \frac {\partial \phi_ {i}}{\partial C _ {i - 1}} \right] \frac {\partial C _ {i - 1}}{\partial \boldsymbol {\theta}}. \tag {7}
$$

This estimator can reduce memory by as much as  $99\%$  without harming optimization; see Figure 6b.

# 6 RELATED WORK

Approximating gradients and matrix operations Much thought has been given to the approximation of general gradients and Jacobians. We draw inspiration from this literature, although our main objective is designing an unbiased gradient estimator, rather than an approximation with bounded accuracy. Abdel-Khalik et al. (2008) accelerate Jacobian accumulation via random projections, in a similar manner to randomized methods for SVD and matrix multiplication. Choromanski & Sindhwani (2017) recover Jacobians in cases where AD is not available by performing a small number of function evaluations with random input perturbations and leveraging known structure of the Jacobian (such as sparsity and symmetry) via compressed sensing.

Other work aims to accelerate neural network training by approximating operations from the forward and/or backward pass. Sun et al. (2017) and Wei et al. (2017) backpropagate sparse gradients, keeping only the top  $k$  elements of the adjoint vector. Adelman & Silberstein (2018) approximate matrix multiplications and convolutions in the forward pass of neural nets nets using a column-row sampling scheme similar to our subsampling scheme. Their method also reduces the computational cost of the backwards pass but changes the objective landscape.

Related are invertible and reversible transformations, which remove the need to save intermediate variables on the forward pass, as these can be recomputed on the backward pass. Maclaurin et al. (2015) use this idea for hyperparameter optimization, reversing the dynamics of SGD with momentum to avoid the expense of saving model parameters at each training iteration. Gomez et al. (2017) introduce a reversible ResNet (He et al., 2016) to avoid storing activations.

Limited-memory learning and optimization Memory is a major bottleneck for reverse-mode AD, and much work aims to reduce its footprint. Gradient checkpointing is perhaps the most well known, and has been used for both reverse-mode AD (Griewank & Walther, 2000) with general layerwise computation graphs, and for neural networks (Chen et al., 2016). In gradient checkpointing, some subset of intermediate variables are saved during function evaluation, and these are used to re-compute downstream variables when required. Gradient checkpointing achieves sublinear memory cost with the number of layers in the computation graph, at the cost of a constant-factor increase in runtime.

Stochastic Computation Graphs Our work is connected to the literature on stochastic estimation of gradients of expected values, or of the expected outcome of a stochastic computation graph. The distinguishing feature of this literature (vs. the proposed RAD approach) is that it uses stochastic estimators of an objective value to derive a stochastic gradient estimator, i.e., the forward pass is randomized. Methods such as REINFORCE (Williams, 1992) optimize an expected return while avoiding enumerating the intractably large space of possible outcomes by providing an unbiased stochastic gradient estimator, i.e., by trading computation for variance. This is also true of mini-batch SGD, and methods for training generative models such as contrastive divergence (Hinton, 2002), and stochastic optimization of evidence lower bounds (Kingma & Welling, 2013). Recent approaches have taken intractable deterministic computation graphs with special structure, i.e. involving loops or the limits of a series of terms, and developed tractable, unbiased, randomized telescoping series-based estimators for the graph's output, which naturally permit tractable unbiased gradient estimation (Tallec & Ollivier, 2017; Beatson & Adams, 2019; Chen et al., 2019; Luo et al., 2020).

# 7 CONCLUSION

We present a framework for randomized automatic differentiation. Using this framework, we construct reduced-memory unbiased estimators for optimization of neural networks and a linear PDE. Future work could develop RAD formulas for new computation graphs, e.g., using randomized rounding to handle arbitrary activation functions and nonlinear transformations, integrating RAD with the adjoint method for PDEs, or exploiting problem-specific sparsity in the Jacobians of physical simulators. The randomized view on AD we introduce may be useful beyond memory savings: we hope it could be a useful tool in developing reduced-computation stochastic gradient methods or achieving tractable optimization of intractable computation graphs.

# REFERENCES

Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pp. 265-283, 2016.  
Hany S Abdel-Khalik, Paul D Hovland, Andrew Lyons, Tracy E Stover, and Jean Utke. A low rank approach to automatic differentiation. In Advances in Automatic Differentiation, pp. 55-65. Springer, 2008.  
Menachem Adelman and Mark Silberstein. Faster neural network training with approximate tensor operations. arXiv preprint arXiv:1805.08079, 2018.  
Friedrich L Bauer. Computational graphs and rounding error. SIAM Journal on Numerical Analysis, 11(1):87-96, 1974.  
Atilim Gunes Baydin, Barak A Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind. Automatic differentiation in machine learning: a survey. Journal of Machine Learning Research, 18(153), 2018.  
Alex Beatson and Ryan P Adams. Efficient optimization of loops and limits with randomized telescoping sums. In International Conference on Machine Learning, 2019.  
James Bergstra, Olivier Breuleux, Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, Guillaume Desjardins, Joseph Turian, David Warde-Farley, and Yoshua Bengio. Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy), volume 4, 2010.  
Christian Bischof, Alan Carle, George Corliss, Andreas Griewank, and Paul Hovland. ADIFOR-- generating derivative codes from Fortran programs. Scientific Programming, 1(1):11-29, 1992.  
Tian Qi Chen, Jens Behrmann, David K Duvenaud, and Jorn-Henrik Jacobsen. Residual flows for invertible generative modeling. In Advances in Neural Information Processing Systems, pp. 9913-9923, 2019.  
Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.  
Krzysztof M Choromanski and Vikas Sindhwani. On blackbox backpropagation and Jacobian sensing. In Advances in Neural Information Processing Systems, pp. 6521-6529, 2017.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-2159, 2011.  
Conal Elliott. The simple essence of automatic differentiation. Proceedings of the ACM on Programming Languages, 2(ICFP):70, 2018.  
Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. In Advances in Neural Information Processing Systems, pp. 2214-2224, 2017.  
A Griewank and U Naumann. Accumulating Jacobians by vertex, edge, or face elimination. cari 2002. In Proceedings of the 6th African Conference on Research in Computer Science, INRIA, France, pp. 375-383, 2002.  
Andreas Griewank and Andrea Walther. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Transactions on Mathematical Software (TOMS), 26(1):19-45, 2000.  
Andreas Griewank and Andrea Walther. Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation, volume 105. SIAM, 2008.  
Laurent Hascoet and Valérie Pascual. The Tapenade automatic differentiation tool: Principles, model, and specification. ACM Transactions on Mathematical Software (TOMS), 39(3):20, 2013.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 2016.  
Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation, 14(8):1771-1800, 2002.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1-42, 1997.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In International Conference on Learning Representations, 2017.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015.  
Yucen Luo, Alex Beatson, Mohammad Norouzi, Jun Zhu, David Duvenaud, Ryan P Adams, and Ricky TQ Chen. Sumo: Unbiased estimation of log marginal probability for latent variable models. In International Conference on Learning Representations, 2020.  
Dougal Maclaurin, David Duvenaud, and Ryan P Adams. Autograd: Effortless gradients in numpy. URL https://github.com/HIPS/autograd.  
Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning, pp. 2113-2122, 2015.  
Ryan G. McClarren. Computational Nuclear Engineering and Radiological Science Using Python: Chapter 18 - One-Group Diffusion Equation. Academic Press, 2018.  
Uwe Naumann. Optimal accumulation of Jacobian matrices by elimination methods on the dual computational graph. Mathematical Programming, 99(3):399-421, 2004.  
Uwe Naumann. Optimal Jacobian accumulation is NP-complete. Mathematical Programming, 112 (2):427-441, 2008.  
Samuel L Smith and Quoc V Le. A Bayesian perspective on generalization and stochastic gradient descent. In International Conference on Learning Representations, 2018.  
Xu Sun, Xuancheng Ren, Shuming Ma, and Houfeng Wang. meprop: Sparsified back propagation for accelerated deep learning with reduced overfitting. In Proceedings of the 34th International Conference on Machine Learning, pp. 3299-3308. JMLR.org, 2017.  
Coretin Tallec and Yann Ollivier. Unbiasing truncated backpropagation through time. arXiv preprint arXiv:1705.08209, 2017.  
Bart van Merrienboer, Dan Moldovan, and Alexander Wiltschko. Tangent: Automatic differentiation using source-code transformation for dynamically typed array programming. In Advances in Neural Information Processing Systems, pp. 6256-6265, 2018.  
Andrea Walther and Andreas Griewank. Getting started with ADOL-C. Combinatorial Scientific Computing, (09061):181-202, 2009.  
Bingzhen Wei, Xu Sun, Xuancheng Ren, and Jingjing Xu. Minimal effort back propagation for convolutional neural networks. arXiv preprint arXiv:1709.05804, 2017.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.
