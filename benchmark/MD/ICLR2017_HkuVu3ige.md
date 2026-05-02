# ON ORTHOGONALITY AND LEARNING RECURRENT NETWORKS WITH LONG TERM DEPENDENCIES

Eugene Vorontsov  $^{1,2}$ , Chiheb Trabelsi  $^{1,2}$ , Samuel Kadoury  $^{1,3}$ , Chris Pal  $^{1,2}$

<sup>1</sup> École Polytechnique de Montréal, Montréal, Canada  
2 Montreal Institute for Learning Algorithms, Montreal, Canada  
<sup>3</sup> CHUM Research Center, Montréal, Canada {eugene.vorontsov, chiheb.trabelsi, samuel.kadoury, christopher.pal}@polymtl.ca

# ABSTRACT

It is well known that it is challenging to train deep neural networks and recurrent neural networks for tasks that exhibit long term dependencies. The vanishing or exploding gradient problem is a well known issue associated with these challenges. One approach to addressing vanishing and exploding gradients is to use either soft or hard constraints on weight matrices so as to encourage or enforce orthogonality. Orthogonal matrices preserve gradient norm during backpropagation and can therefore be a desirable property; however, we find that hard constraints on orthogonality can negatively affect the speed of convergence and model performance. This paper explores the issues of optimization convergence, speed and gradient stability using a variety of different methods for encouraging or enforcing orthogonality. In particular we propose a weight matrix factorization and parameterization strategy through which we can bound matrix norms and therein control the degree of expansivity induced during backpropagation.

# 1 INTRODUCTION

The depth of deep neural networks confers representational power, but also makes model optimization more challenging. Training deep networks with gradient descent based methods is known to be difficult as a consequence of the vanishing and exploding gradient problem (Hochreiter & Schmidhuber, 1997). Typically, exploding gradients are avoided by clipping large gradients (Pascanu et al., 2013) or introducing an  $L_{2}$  or  $L_{1}$  weight norm penalty. The latter has the effect of bounding the spectral radius of the linear transformations, thus limiting the maximal gain across the transformation. Krueger & Memisevic (2015) attempt to stabilize the norm of propagating signals directly by penalizing differences in successive norm pairs in the forward pass and Pascanu et al. (2013) propose to penalize successive gradient norm pairs in the backward pass. These regularizers affects the network parameterization with respect to the data instead of penalizing weights directly.

Both expansivity and contractivity of linear transformations can also be limited by more tightly bounding their spectra. By limiting the transformations to be orthogonal, their singular spectra are limited to unitary gain causing the transformations to be norm-preserving. Le et al. (2015) and Henaff et al. (2016) have respectively shown that identity initialization and orthogonal initialization can be beneficial. Arjovsky et al. (2015) have gone beyond initialization, building unitary RNN models with transformations that are unitary by construction which they achieved by composing multiple basic unitary transformations. The resulting transformations, for some n-dimensional input, cover only some subset of possible  $n \times n$  unitary matrices but appear to perform well on simple tasks and have the benefit of having low complexity in memory and computation.

The entire set of possible unitary or orthogonal parameterizations forms the Stiefel manifold. At a much higher computational cost, gradient descent optimization directly along this manifold can be done via geodesic steps (Nishimori, 2005; Tagare, 2011). Recent work (Wisdom et al., 2016) has proposed the optimization of unitary matrices along the Stiefel manifold using geodesic gradient descent. To produce a full-capacity parameterization for unitary matrices they use some insights from Tagare (2011) combining the use of a canonical inner products and Cayley transformations.

Their experimental work indicates that full capacity unitary RNN models can solve the copy memory problem whereas both LSTM networks and restricted capacity unitary RNN models having similar complexity appear unable to solve the task for longer sequence length ( $T = 2000$ ).

In contrast, here we explore the optimization of real valued matrices within a configurable margin about the Stiefel manifold. We suspect that a strong constraint of orthogonality limits the model's representational power, hindering its performance, and may make optimization more difficult. We explore this hypothesis empirically by employing a factorization technique that allows us to limit the degree of deviation from the Stiefel manifold. While we use geodesic gradient descent, we simultaneously update the singular spectra of our matrices along Euclidean steps, allowing optimization to step away from the manifold while still curving about it.

# 1.1 VANISHING AND EXPLODING GRADIENTS

The issue of vanishing and exploding gradients as it pertains to the parameterization of neural networks can be illuminated by looking at the gradient back-propagation chain through a network.

A neural network with  $n$  hidden layers has pre-activations

$$
\mathbf {a} _ {i} \left(\mathbf {h} _ {i - 1}\right) = \mathbf {W} _ {i} \mathbf {h} _ {i - 1} + \mathbf {b} _ {i}, i \in \{2, \dots , n \} \tag {1}
$$

For notational convenience, we combine parameters  $\mathbf{W}_i$  and  $\mathbf{b}_i$  to form an affine matrix  $\theta$ . We can see that for some loss function  $L$  at layer  $n$ , the derivative with respect to parameters  $\theta_i$  is:

$$
\frac {\partial L}{\partial \boldsymbol {\theta} _ {i}} = \frac {\partial \mathbf {a} _ {n + 1}}{\partial \boldsymbol {\theta} _ {i}} \frac {\partial L}{\partial \mathbf {a} _ {n + 1}} \tag {2}
$$

The partial derivatives for the pre-activations can be decomposed as follows:

$$
\begin{array}{l} \frac {\partial \mathbf {a} _ {i + 1}}{\partial \boldsymbol {\theta} _ {i}} = \frac {\partial \mathbf {a} _ {i}}{\partial \boldsymbol {\theta} _ {i}} \frac {\partial \mathbf {h} _ {i}}{\partial \mathbf {a} _ {i}} \frac {\partial \mathbf {a} _ {i + 1}}{\partial \mathbf {h} _ {i}} \\ = \frac {\partial \mathbf {a} _ {i}}{\partial \boldsymbol {\theta} _ {i}} \mathbf {D} _ {i} \mathbf {W} _ {i + 1} \rightarrow \frac {\partial \mathbf {a} _ {i + 1}}{\partial \mathbf {a} _ {i}} = \mathbf {D} _ {i} \mathbf {W} _ {i + 1}, \\ \end{array}
$$

where  $\mathbf{D_i}$  is the Jacobian corresponding to the activation function, containing partial derivatives of the hidden units at layer  $i + 1$  with respect to the pre-activation inputs. Typically,  $\mathbf{D}$  is diagonal. Following the above, the gradient in equation 2 can be fully decomposed into a recursive chain of matrix products:

$$
\frac {\partial L}{\partial \boldsymbol {\theta} _ {i}} = \frac {\partial \mathbf {a} _ {i}}{\partial \boldsymbol {\theta} _ {i}} \prod_ {j = i} ^ {n} \left(\mathbf {D} _ {j} \mathbf {W} _ {j + 1}\right) \frac {\partial L}{\partial \mathbf {a} _ {n + 1}} \tag {4}
$$

In (Pascanu et al., 2013), it is shown that the 2-norm of  $\frac{\partial\mathbf{a}_{i + 1}}{\partial\mathbf{a}_i}$  is bounded by the product of the norms of the non-linearity's Jacobian and transition matrix at time  $t$  (layer  $i$ ), as follows:

$$
\left| \left| \frac {\partial \mathbf {a} _ {t + 1}}{\partial \mathbf {a} _ {t}} \right| \right| \leq | | \mathbf {D} _ {t} | | | | \mathbf {W} _ {t} | | \leq \lambda_ {\mathbf {D} _ {t}} \lambda_ {\mathbf {W} _ {t}} = \eta_ {t}, \tag {5}
$$

$$
\lambda_ {\mathbf {D} _ {t}}, \lambda_ {\mathbf {W} _ {t}} \in \mathbb {R}.
$$

where  $\lambda_{\mathbf{D}_t}$  and  $\lambda_{\mathbf{W}_t}$  are the largest singular values of the non-linearity's Jacobian  $\mathbf{D}_t$  and the transition matrix  $\mathbf{W}_t$ . In RNNs,  $\mathbf{W}_t$  is shared across time and can be simply denoted as  $\mathbf{W}$ .

Equation 5 shows that the gradient can grow or shrink at each layer depending on the gain of each layer's linear transformation  $\mathbf{W}$  and the gain of the Jacobian  $\mathbf{D}$ . The gain caused by each layer is magnified across all time steps or layers. It is easy to have extreme amplification in a recurrent neural network where  $\mathbf{W}$  is shared across time steps and a non-unitary gain in  $\mathbf{W}$  is amplified exponentially. The phenomena of extreme growth or contraction of the gradient across time steps or layers are known as the exploding and the vanishing gradient problems, respectively. It is sufficient for RNNs to have  $\eta_t \leq 1$  at each time  $t$  to enable the possibility of vanishing gradients, typically for some large number of time steps  $T$ . The rate at which a gradient (or forward signal) vanishes depends on both the parameterization of the model and on the input data. The parameterization

may be conditioned by placing appropriate constraints on  $\mathbf{W}$ . It is worth keeping in mind that the Jacobian  $\mathbf{D}$  is typically contractive, thus tending to be norm-reducing) and is also data-dependent, whereas  $\mathbf{W}$  can vary from being contractive to norm-preserving, to expansive and applies the same gain on the forward signal as on the back-propagated gradient signal.

# 2 OUR APPROACH

Vanishing and exploding gradients can be controlled to a large extent by controlling the maximum and minimum gain of  $\mathbf{W}$ . The maximum gain of a matrix  $\mathbf{W}$  is given by the spectral norm which is given by

$$
\left| \left| \mathbf {W} \right| \right| _ {2} = \max  \left[ \frac {\left| \left| \mathbf {W} \mathbf {x} \right| \right|}{\left| \left| \mathbf {x} \right| \right|} \right]. \tag {6}
$$

By keeping our weight matrix  $\mathbf{W}$  close to orthogonal, one can ensure that it is close to a norm-preserving transformation (where the spectral norm is equal to one, but the minimum gain is also one). One way to achieve this is via a simple soft constraint or regularization term of the form:

$$
\lambda \sum_ {i} \left\| \mathbf {W} _ {i} ^ {T} \mathbf {W} _ {i} - \mathbf {I} \right\| ^ {2}. \tag {7}
$$

However, it is possible to formulate a more direct parameterization or factorization for  $\mathbf{W}$  which permits hard bounds on the amount of expansion and contraction induced by  $\mathbf{W}$ . This can be achieved by simply parameterizing  $\mathbf{W}$  according to its singular value decomposition, which consists of the composition of orthogonal basis matrices  $\mathbf{U}$  and  $\mathbf{V}$  with a diagonal spectral matrix  $\mathbf{S}$  containing the singular values which are real and positive by definition. We have

$$
\mathbf {W} = \mathbf {U S V} ^ {T}. \tag {8}
$$

Since the spectral norm or maximum gain of a matrix is equal to its largest singular value, this decomposition allows us to control the maximum gain or expansivity of the weight matrix by controlling the magnitude of the largest singular value. Similarly, the minimum gain or contractivity of a matrix can be obtained from the minimum singular value.

We can keep the bases  $\mathbf{U}$  and  $\mathbf{V}$  orthogonal via geodesic gradient descent along the set of weights that satisfy  $\mathbf{U}^T\mathbf{U} = \mathbf{I}$  and  $\mathbf{V}^T\mathbf{V} = \mathbf{I}$  respectively. The submanifolds that satisfy these constraints are called Stiefel manifolds. We discuss how this is achieved in more detail below, then discuss our construction for bounding the singular values.

During optimization, in order to maintain the orthogonality of an orthogonally-initialized matrix  $\mathbf{M}$ , i.e. where  $\mathbf{M} = \mathbf{U}$ ,  $\mathbf{M} = \mathbf{V}$  or  $\mathbf{M} = \mathbf{W}$  if so desired, we employ a Cayley transformation of the update step onto the Stiefel manifold of (semi-)orthogonal matrices, as in Nishimori (2005) and Tagare (2011). Given an orthogonally-initialized parameter matrix  $\mathbf{M}$  and its Jacobian,  $\mathbf{G}$  with respect to the objective function, an update is performed as follows:

$$
\mathbf {A} = \mathbf {G M} ^ {T} - \mathbf {M G} ^ {T}
$$

$$
\mathbf {M} _ {\text {n e w}} = \mathbf {M} + (\mathbf {I} + \frac {\eta}{2} \mathbf {A}) ^ {- 1} (\mathbf {I} - \frac {\eta}{2} \mathbf {A}), \tag {9}
$$

where  $\mathbf{A}$  is a skew-symmetric matrix (that depends on the Jacobian and on the parameter matrix) which is mapped to an orthogonal matrix via a Cayley transform and  $\eta$  is the learning rate.

While the update rule in (9) allows us to maintain an orthogonal hidden to hidden transition matrix  $\mathbf{W}$  if desired, we are interested in exploring the effect of stepping away from the Stiefel manifold. As such, we parameterize the transition matrix  $\mathbf{W}$  in factorized form, as a singular value decomposition with orthogonal bases  $\mathbf{U}$  and  $\mathbf{V}$  updated by geodesic gradient descent using the Cayley transform approach above.

If  $\mathbf{W}$  is an orthogonal matrix, the singular values in the diagonal matrix  $\mathbf{S}$  are all equal to one. However, in our formulation we allow these singular values to deviate from one and employ a sigmoidal parameterization to apply a hard constraint on the maximum and minimum amount of deviation. Specifically, we define a margin  $m$  around 1 within which the singular values must lie. This is achieved with the parameterization

$$
s _ {i} = 2 m \left(\sigma \left(p _ {i}\right) - 0. 5\right) + 1, \quad s _ {i} \in \{\operatorname {d i a g} (\mathbf {S}) \}, m \in [ 0, 1 ]. \tag {10}
$$

The singular values are thus restricted to the range  $[1 - m, 1 + m]$  and the underlying parameters  $p_i$  are updated freely via stochastic gradient descent. Note that this parameterization strategy also has implications on the step sizes that gradient descent based optimization will take when updating the singular values – they tend to be smaller compared to models with no margin constraining their values. Specifically, a singular value's progression toward a margin is slowed the closer it is to the margin. The sigmoidal parameterization can also impart another effect on the step size along the spectrum which needs to be accounted for. Considering 10, the gradient backpropagation of some loss  $L$  toward parameters  $p_i$  is found as

$$
\frac {d L}{d p _ {i}} = \frac {d s _ {i}}{d p _ {i}} \frac {d L}{d s _ {i}} = 2 m \frac {d \sigma (p _ {i})}{d p _ {i}} \frac {d L}{d s _ {i}}. \tag {11}
$$

From (11), it can be seen that the magnitude of the update step for  $p_i$  is scaled by the margin hyperparameter  $m$ . This means for example that for margins less than one, the effective learning rate for the spectrum is reduced in proportion to the margin. Consequently, we adjust the learning rate along the spectrum to be independent of the margin by renormalizing it by  $2m$ .

This margin formulation both guarantees singular values lie within a well defined range and slows deviation from orthogonality. Alternatively, one could enforce the orthogonality of  $\mathbf{U}$  and  $\mathbf{V}$  and impose a regularization term corresponding to a mean one Gaussian prior on these singular values. This encourages the weight matrix  $\mathbf{W}$  to be norm preserving with a controllable strength equivalent to the variance of the Gaussian. We also explore this approach further below.

# 3 EXPERIMENTS

In this section, we explore hard and soft orthogonality constraints on factorized weight matrices for recurrent neural network hidden to hidden transitions. With hard orthogonality constraints on  $\mathbf{U}$  and  $\mathbf{V}$ , we investigate the effect of widening the spectral margin or bounds on convergence and performance. Loosening these bounds allows increasingly larger margins within which the transition matrix  $\mathbf{W}$  can deviate from orthogonality. We confirm that orthogonal initialization is useful as noted in Henaff et al. (2016), and we show that although strict orthogonality guarantees stable gradient norm, loosening orthogonality constraints can increase the rate of gradient descent convergence. We perform our analyses on a sequence copying task (Hochreiter & Schmidhuber, 1997) and classification tasks based on sequential and permuted MNIST vectors (Le et al., 2015; LeCun et al., 1998), tasks which are designed so as to require models to capture long-range dependencies.

# THE COPY TASK

The copy task, introduced by Hochreiter & Schmidhuber (1997), is a synthetic benchmark with pathologically hard long distance dependencies. In our use, it consists of an input sequence that must be remembered by the network, followed by a series of blank inputs terminated by a delimiter that denotes the point at which the network must begin to output a copy of the initial sequence. We use an input sequence of  $T + 20$  elements that begins with a sub-sequence of 10 elements to copy, each containing a symbol  $a_{i} \in \{a_{1},\dots,a_{p}\}$  out of  $p = 8$  possible symbols. This sub-sequence is followed by  $T - 1$  elements of the blank category  $a_{0}$  which is terminated at step  $T$  by a delimiter symbol  $a_{p + 1}$  and 10 more elements of the blank category. The network must learn to remember the initial 10 element sequence for  $T$  time steps and output it after receiving the delimiter symbol.

# SEQUENTIAL MNIST AND PERMUTEDSEQUENTIAL MNIST

The sequential MNIST task from Le et al. (2015), MNIST digits are flattened into vectors that can be traversed sequentially by a recurrent neural network. The goal is to classify the digit based on the sequential input of pixels. The simple variant of this task is with a simple flattening of the image matrices; the harder variant of this task includes a random permutation of the pixels in the input vector that is determined once for an experiment. The latter formulation introduces longer distance dependencies between pixels that must be interpreted by the classification model.

# EXPLORATION

Below we explore the effect of loosening hard orthogonality constraints through loosening the spectral margin defined above for the hidden to hidden transition matrix on a copy task and for sequential MNIST digit classification.

In all experiments, we employed RMSprop (Tieleman & Hinton, 2012) when not using geodesic gradient descent. We used minibatches of size 50 and for generated data (the copy task), we assumed an epoch length of 100 minibatches. We cautiously introduced gradient clipping at magnitude 100 (unless stated otherwise) in all of our RNN experiments although it may not always be required and we consistently applied a small weight decay of 0.0001. When parameterizing a matrix in factorized form, we apply the weight decay on the composite matrix rather than on the factors in order to be consistent across experiments.

For different sequence lengths  $T$ , we trained simple recurrent neural networks with the hidden to hidden matrix factorization as in (8) and various spectral margins  $m$  in (10). We set the learning rate at  $1 \times 10^{-6}$  for geodesic gradient descent along the bases  $\mathbf{U}$  and  $\mathbf{V}$  and  $1 \times 10^{-4}$  for RMSprop on the remaining parameters. For the copy task, we used Elman networks without a transition non-linearity as in Henaff et al. (2016); we found that non-linearities such as a rectified linear unit (ReLU) (Nair & Hinton, 2010) or hyperbolic tangent (tanh) made this task far more difficult to solve. In fact, when using transition non-linearities, we could only solve the task when using orthogonality constraints as we found an orthogonal initialization alone to be insufficient in that case even for short sequence lengths ( $T = 100$ ) and small learning rates. It is worth noting that in the unitary evolution recurrent neural network of Arjovsky et al. (2015), the non-linearity (referred to as the "modReLU") is actually initialized as an identity operation that may deviate from identity during training.

![](images/7a97f788e6c3eb0cbce725509160385242acb800d5c52ee867e94fecdbe51a8d.jpg)  
Figure 1: Accuracy curves on the copy task for sequence lengths of (from left to right)  $T = 200$ ,  $T = 500$ ,  $T = 1000$ ,  $T = 10000$  given different spectral margins. Convergence speed increases with margin size; however, large margin sizes are ineffective at longer sequence lengths ( $T = 10000$ , right).

![](images/c148447ddca3e729706680b83ff66af77c73053a9ea2bd0232bf5974df3f8839.jpg)

![](images/b0b36de16d4c0c89066ce048390b040e189670717833f6967b2bacaf9916f1ff.jpg)

![](images/45c592a66ecc4ddf265ed197958c6296c8a7cb99763143bdf29d490a4d9d8d30.jpg)

As shown in Figure 1 we see an increase in the rate of convergence as we increase the spectral margin. This observation generally holds across the tested sequence lengths ( $T = 200$ ,  $T = 500$ ,  $T = 1000$ ,  $T = 10000$ ); however, large spectral margins hinder convergence on extremely long sequence lengths. At sequence length  $T = 10000$ , parameterizations with spectral margins larger than 0.001 converge slower than when using a margin of 0.001. In addition, the experiment without a margin failed to converge on the longest sequence length. This follows the expected pattern where stepping away from the Stiefel manifold may help with gradient descent optimization but loosening orthogonality constraints can reduce the stability of signal propagation through the network.

Although we can explore convergence on the copy task, we cannot otherwise investigate model performance on the copy task as converging models converge to perfect accuracy. We show the results of experiments on permuted sequential MNIST in Table 2 and ordered sequential MNIST in Table 1. The loss curves can be seen in Figure 2. We trained the factorized RNN models for 120 epochs with geodesic gradient descent on the bases and RMSprop on the spectrum (both with learning rate 0.001), using a tanh transition nonlinearity, and clipping gradients of 100 magnitude. We also trained an LSTM with 128 hidden units on both tasks for 150 epochs, configured with peephole connections, orthogonally initialized (and bias initialized to one), and trained with RMSprop (learning rate 0.0001, clipping gradients of magnitude 1).

Interestingly in both the ordered and permuted sequential MNIST tasks, the optimal margin appears to be near 0.1. Thus, similar to the longer sequence copy tasks examined in Figure 1, the optimal margin does not appear to lie at the extremes of zero or one. In all cases, models with a non-zero margin significantly outperform those that are constrained to be purely orthogonal (margin of zero) and those that are initialized with Glorot normal initialization (Glorot & Bengio, 2010), without spectral

<table><tr><td>margin</td><td>initialization</td><td>accuracy</td></tr><tr><td>0</td><td>orthogonal</td><td>77.18</td></tr><tr><td>0.001</td><td>orthogonal</td><td>79.26</td></tr><tr><td>0.01</td><td>orthogonal</td><td>85.47</td></tr><tr><td>0.1</td><td>orthogonal</td><td>94.10</td></tr><tr><td>1</td><td>orthogonal</td><td>93.84</td></tr><tr><td>none</td><td>orthogonal</td><td>93.24</td></tr><tr><td>none</td><td>Glorot normal</td><td>66.71</td></tr><tr><td colspan="2">LSTM</td><td>97.30</td></tr></table>

Table 1: Ordered sequential MNIST with different margin sizes and an LSTM.  

<table><tr><td>margin</td><td>initialization</td><td>accuracy</td></tr><tr><td>0</td><td>orthogonal</td><td>83.56</td></tr><tr><td>0.001</td><td>orthogonal</td><td>84.59</td></tr><tr><td>0.01</td><td>orthogonal</td><td>89.63</td></tr><tr><td>0.1</td><td>orthogonal</td><td>91.44</td></tr><tr><td>1</td><td>orthogonal</td><td>90.83</td></tr><tr><td>none</td><td>orthogonal</td><td>90.51</td></tr><tr><td>none</td><td>Glorot normal</td><td>79.33</td></tr><tr><td colspan="2">LSTM</td><td>92.62</td></tr></table>

Table 2: Permuted sequential MNIST with different margin sizes and an LSTM.

![](images/22c4b97f77a3e0f0ed0b6c53c83eeddd4abb25257825e954b07ad081b89f6d7f.jpg)  
Figure 2: Loss curves for different factorized RNN parameterizations on the sequential MNIST task (left) and the permuted sequential MNIST task (right). The spectral margin is denoted by m; models with no margin have singular values that are directly optimized with no constraints; Glorot refers to a factorized RNN with no margin that is initialized with Glorot normal initialization.

![](images/8680da17d1b6c27c4b2bb3ddfc4b8247b0871732942f8db433eb986ef11bde4c.jpg)

bounds. This suggests that loosening an orthogonality constraint may increase the representational power of a given parameterization. The best results on both the ordered and sequential MNIST tasks were yielded by models with a spectral margin of 0.1, at  $94.10\%$  accuracy and  $91.44\%$  accuracy, respectively. An LSTM outperformed the RNNs in both tasks; nevertheless, RNNs with hidden to hidden transitions initialized as orthogonal matrices performed admirably without a memory component and without all of the additional parameters associated with gates. Indeed, orthogonally initialized RNNs performed almost on par with the LSTM in the permuted sequential MNIST task which presents longer distance dependencies than the ordered task. Curiously, larger margins and even models without sigmoidal constraints on the spectrum (no margin) performed well as long as they were initialized to be orthogonal suggesting that evolution away from orthogonality is not a serious problem on this task.

It is interesting to note that even long sequence lengths (T=1000) in the copy task can be solved efficiently with rather large margins on the spectrum. In fact, if one were to expect singular values to tend to reach these margins, one would expect to encounter vanishing or exploding gradients. In Figure 3 we look at the gradient propagation of the loss of the last time step in the network with respect to the hidden activations. We can see that for a purely orthogonal parameterization of the transition matrix (when the margin is zero), the gradient norm is preserved across time steps, as expected. We note that a purely orthogonal non-factorized parameterization (RNN optimized with geodesic SGD) performed similarly to the one with a margin of zero (not shown); we chose to compare our experiments to the latter as it is most similar to and has the same number of parameters (not counting the constant spectrum vector) as the other factorized networks. We further observe that with increasing margin size, the number of update steps over which this norm preservation survives decreases. However, as stated above, one may expect this loss of norm-preserving behaviour to happen sooner.

Although the deviation of singular values from one should be slowed by the sigmoidal parameterizations, even parameterizations without a sigmoid (no margin) can be effectively trained for all but the longest sequence lengths. This suggests that the spectrum is not deviating far from orthogonality and that inputs to the hidden to hidden transitions are mostly not aligned along the dimensions of great-

![](images/675eabe5d37b988bbc77f96baf12af2ffd598002db8dae8c7c7797115762b5b0.jpg)  
Figure 3: The norm of the gradient of the loss from the last time step with respect to the hidden units at a given time step for a length 220 RNN over 1000 update iterations for different margins. Iterations are along the abscissa and time steps are denoted along the ordinate. The first column margins are: 0, 0.001, 0.01. The second column margins are: 0.1, 1, no margin. Gradient norms are normalized across the time dimension.

![](images/7ea03ce9fe7c009889907e0dfe8196707f9feb3b764db9ff4986ff7907ae8306.jpg)

est expansion or contraction. We evaluated the spread of the spectrum in all of our experiments and found that indeed, singular values tend to stay well within their prescribed bounds and only reach the margin when using a very large learning rate that does not permit convergence. Furthermore, even without a sigmoidal margin singular values remain near one throughout training for matrices initialized as orthogonal. Interestingly, singular values spread out less for longer sequence lengths (nevertheless, the  $T = 10000$  copy task could not be solved with no sigmoid on the spectrum).

![](images/70c4ba92aa370acee48e84d9cbc043be148c30443655acc3025fb857e164f514.jpg)

![](images/7e97047c8f3c9312eadb8f8ae943feb5953b797b1d0dd1421fc63530ed83c316.jpg)

![](images/e512bfc4b533705e09f34336c631b8db97593fcdeef53319493c06e4d29c3fa4.jpg)

![](images/4929943bfc22ae305e0eeb7c6e7d69e8dcd26643f4795cbbbad8fd85e701c28d.jpg)  
Figure 4: Singular value evolution on the permuted sequential MNIST task for factorized RNNs with different margin sizes. Margins are, from left to right: top row: 0.001, 0.01, 0.1; bottom row: 1, no margin, no margin. The singular value distributions are summarized with the mean (green line, center) and standard deviation (green shading about mean), minimum (red, bottom) and maximum (blue, top) values. All models are initialized with orthogonal hidden to hidden transition matrices except for the model on the bottom right where Glorot normal initialization is used.

![](images/a06b86fbf1139a4179488ec556a4ef92c2d4e2328c9b7a25cb84a7c7b7e322ff.jpg)

![](images/b4d0c163aed6fea5b4014905550b8618a9034d680b3b77c3f6d8e052c84d659f.jpg)

We visualize the spread of singular values for different model parameterizations on the permuted sequential MNIST task in Figure 4. Curiously, we find that the distribution of singular values tended to shift upward to a mean of approximately 1.05 on both the ordered and permuted sequential MNIST tasks. We note that in those experiments, a tanh transition nonlinearity was used which is contractive in both the forward signal pass and the gradient backward pass. An upward shift in the distribution of singular values of the transition matrix would help compensate for that contraction. Indeed, (Saxe et al., 2013) describe this as a possibly good regime for learning in deep neural networks. That the model appears to evolve toward this regime suggests that deviating from it may incur a cost. This is interesting because the cost function cannot take into account numerical issues such as vanishing or exploding gradients (or forward signals); we do not know what could make this deviation costly. Unlike orthogonally initialized models, the RNN with Glorot normal initialized transition matrices,

on the bottom right of Figure 4, begins and ends with a wide singular spectrum. While there is no clear positive shift in the distribution of singular values, the mean value appears to very gradually increase for both the ordered and permuted sequential MNIST tasks. If the model is to be expected to positively shift singular values to compensate for the contractivity of the tanh nonlinearity, it is not doing so well for the Glorot-initialized case; however, this may be due to the inefficiency of training as a result of vanishing gradients, given that initialization.

# 3.1 OTHER SOFT ORTHOGONALITY CONSTRAINT APPROACHES

Having established that it may indeed be useful to step away from orthogonality, here we explore two forms of soft constraints (rather than hard bounds as above) on hidden to hidden transition matrix orthogonality. The first is a simple penalty that directly encourages a transition matrix  $\mathbf{W}$  to be orthogonal, of the form  $\lambda ||\mathbf{W}^T\mathbf{W} - \mathbf{I}||_2^2$ . This is similar to the orthogonality penalty introduced by Henaff et al. (2016). In the first two subfigures on the left of Figure 5, we explore the effect of weakening this form of regularization. We trained both a regular non-factorized RNN on the  $T = 200$  copy task and a factorized RNN with orthogonal bases on the  $T = 500$  copy task. For the regular RNN, we had to reduce the learning rate to  $10^{-5}$ . Here again we see that weakening the strength of the orthogonality-encouraging penalty can increase convergence speed.

![](images/4a28052a6956e7dbcc181fd05357bed95890227b9ad24f663307cbb2300a63dc.jpg)  
Figure 5: Accuracy curves on the copy task for different strengths of soft orthogonality constraints. A soft orthogonality constraint is applied to the transition matrix  $\mathbf{W}$  for a regular RNN on  $T = 200$  (Left) and the same is applied on a factorized RNN on  $T = 500$  (Left center). Another constraint in the form of a mean one Gaussian prior on the singular values is applied to a factorized RNN on  $T = 200$  (Right center); the same is applied to a factorized RNN with a sigmoidal parameterization of the spectrum, using a large margin of 1 (Right). Loosening orthogonality speeds convergence.

![](images/86f619ad21c8e5a7b85fd7b90ef8bf193c7ea93aa6adaadf365e6ac2a80965d1.jpg)

![](images/726efa7e6f3989a74b039fbdda658c597fa4c62e0ff8c4fd97c2c80bcd9efd7c.jpg)

![](images/b559148db41470cd80b7bd431eeebd12dfbd0b2c7a8a17518b437fe6ee14303f.jpg)

The second approach we explore replaces the sigmoidal margin parameterization with a mean one Gaussian prior on the singular values. In the two right subfigures of Figure 5, we visualize the accuracy on the length 200 copy task, using geoSGD (learning rate  $10^{-6}$ ) to keep  $\mathbf{U}$  and  $\mathbf{V}$  orthogonal and different strengths of a Gaussian prior with mean one on the singular values. We trained these experiments with regular SGD on the spectrum, using a  $10^{-5}$  learning rate. We see that priors which are too strong lead to slow convergence. Loosening the strength of the prior makes the optimization more efficient. Furthermore, we compare a direct parameterization of the spectrum (no sigmoid) in Figure 5 with a sigmoidal parameterization, using a large margin of 1. Without the sigmoidal parameterization, optimization quickly becomes unstable; on the other hand, the optimization also becomes unstable if the prior is removed completely in the sigmoidal formulation (margin 1). These results further motivate the idea that parameterizations that deviate from orthogonality may perform better than purely orthogonal ones, as long as they are sufficiently constrained to avoid instability during training.

# 4 CONCLUSIONS AND DISCUSSION

We have explored a number of methods for controlling the expansivity of gradients during backpropagation based learning in RNNs through manipulating orthogonality constraints and regularization on matrices. Our experiments indicate that moving away from hard constraints on matrix orthogonality can help optimization and model performance. However, we observe consistently that relaxing regularization which encourages the spectral norms of weight matrices to be close to one, or allowing bounds on the spectral norms of weight matrices to be too wide, reverses these gains and can lead to unstable optimization.

# ACKNOWLEDGMENTS

We thank the Natural Sciences and Engineering Research Council (NSERC) of Canada and Samsung for supporting this research.

# REFERENCES

Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Aistats, volume 9, pp. 249-256, 2010.  
Mikael Henaff, Arthur Szlam, and Yann LeCun. Orthogonal rnns and long-memory tasks. arXiv preprint arXiv:1602.06662, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
David Krueger and Roland Memisevic. Regularizing rnns by stabilizing activations. arXiv preprint arXiv:1511.08400, 2015.  
Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. arXiv preprint arXiv:1504.00941, 2015.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pp. 807-814, 2010.  
Yasunori Nishimori. A note on riemannian optimization methods on the stiefel and the grassmann manifolds. dim, 1:2, 2005.  
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. ICML (3), 28:1310-1318, 2013.  
Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.  
Hemant D Tagare. Notes on optimization on stiefel manifolds. Technical report, Tech. Rep., Yale University, 2011.  
T. Tieleman and G. Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.  
Scott Wisdom, Thomas Powers, John R. Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity unitary recurrent neural networks. To appear in NIPS, 2016.