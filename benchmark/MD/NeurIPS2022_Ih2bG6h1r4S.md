# Atlas: Universal Function Approximator For Memory Retention

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Artificial neural networks (ANNs), despite their universal function approximation capability and practical success, are subject to catastrophic forgetting. Catastrophic forgetting refers to the abrupt unlearning of a previous task when a new task is learned. It is an emergent phenomenon that plagues ANNs and hinders continual learning. Existing universal function approximation theorems for ANNs guarantee function approximation ability, but seldom touch on the model details and do not predict catastrophic forgetting. This paper presents a novel universal approximation theorem for multi-variable functions using only single-variable functions and exponential functions. Furthermore, we present Atlas—a novel ANN architecture based on the exponential approximation theorem and B-splines. It is shown that Atlas is a universal function approximator capable of memory retention and, therefore, continual learning. The memory retention of Atlas is imperfect, with some off-target effects during continual learning, but it is well-behaved and predictable. An efficient implementation of Atlas is provided. Experiments are conducted to evaluate both the function approximation and memory retention capabilities of Atlas.

# 1 Introduction

Universal function approximation theorems are a cornerstone of machine learning, and prove that artificial neural networks (ANNs) can approximate any given continuous target function with arbitrarily small error [6, 7, 11]. The theorems do not specify how to find a given ANN architecture and parameters with sufficient performance for problems in practice. Another problem that arises in practice is catastrophic forgetting [3, 9, 15], where an ANN would learn a new task and the subsequent parameter updates would interfere with the model's performance on previously learned tasks. Catastrophic forgetting is also called catastrophic interference [13]. Catastrophic forgetting is like learning to pick up a cup, but simultaneously forgetting how to breathe.

If an ANN cannot effectively learn many tasks, it has limited utility in the context of continual learning [4, 8]. ANNs have other practical problems, such as vanishing or exploding gradients [5], which make training unstable. It is also not obvious how to increase the size of a trained ANN to better fit to data without damaging model performance and having to retrain the model on already seen data.

This paper introduces Atlas—a novel universal function approximator based on B-splines that has intrinsic memory retention. Atlas has well-behaved parameter gradients that do not vanish or explode. Atlas has methods to expand model capacity in a systematic way without loosing the previously learned information. The accompanying representation and universal approximation theorems are also provided.

# 2 Relevant Studies

It is conjectured that overlapping representations in ANNs lead to catastrophic forgetting [8]. Catastrophic forgetting occurs when parameters necessary/needed for one task change while training to meet the objectives of another task [10, 14]. Strategies to mitigate catastrophic forgetting can include shielding important weights from changes, or regularisation to keep weights close to their initial values. The least desirable option is retraining a model over all tasks, again. Data augmentation approaches like rehearsal and pseudo-rehearsal have also been employed [15].

Pi-sigma neural networks use nodes that compute products instead of sums [17]. Pi-sigma neural networks reportedly have potential utility in building more efficient and expressive models than the standard ANNs. Atlas can similarly compute products, and is more easily trained than the Pi-sigma neural networks.

B-splines, which form the basis of Atlas, have been applied for machine learning [2]. Scardapane et al. [16] investigated trainable activation functions parameterised by splines. Uniform cubic B-splines have basis functions that are translates of one another [1]. Uniform cubic B-splines have been tested for memory retention, and Atlas can be considered as an improvement on existing spline additive models [18].

# 3 Exponential Representation Theorem

Any continuous multi-variable function on a compact space can be uniformly approximated with multi-variable polynomials by the Stone-Weierstrass Theorem. Let  $\mathcal{I}$  denote an index set of tuples of natural numbers including zero such that  $i_j \in \mathbb{N}^0$  for all  $j \in \mathbb{N}$  with  $i = (i_1,.., i_n) \in \mathcal{I}$  and  $a_i \in \mathbb{R}$ . Multi-variable polynomials can be represented as:

$$
y (\vec {\mathbf {x}}) = y (x _ {1},.., x _ {n}) = \sum_ {i \in \mathcal {I}} a _ {i} x _ {1} ^ {i _ {1}} x _ {2} ^ {i _ {2}}... x _ {n} ^ {i _ {n}} = \sum_ {i \in \mathcal {I}} a _ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}}
$$

Each monomial term  $a_{i}\Pi_{j = 1}^{n}x_{j}^{x_{j}}$  is a product of single-variable functions in each variable. It is desirable to rewrite products as sums. The product of single-variable functions can be rewritten as a sum using exponentials and logarithms.

Lemma 1. For any  $a_i \in \mathbb{R}$ , there exists  $\gamma_i > 0$  and  $\beta_i > 0$ , such that:  $a_i = \gamma_i - \beta_i$

Theorem 1 (Exponential representation theorem). Any multi-variable polynomial function  $y(\vec{\mathbf{x}})$  of  $n$  variables over the positive orthant, can be exactly represented by continuous single-variable functions  $g_{i,j}(x_j)$  and  $h_{i,j}(x_j)$  in the form:

$$
y (\vec {\mathbf {x}}) = \sum_ {i \in \mathcal {I}} \exp \left(\Sigma_ {j = 1} ^ {n} g _ {i, j} (x _ {j})\right) - \exp \left(\Sigma_ {j = 1} ^ {n} h _ {i, j} (x _ {j})\right)
$$

Proof. Consider any monomial term  $a_i \Pi_{j=1}^n x_j^{i_j}$  with  $a_i \in \mathbb{R}$ , then by Lemma 1 there exist strictly positive numbers  $\gamma_i > 0$  and  $\beta_i > 0$ , such that:

$$
\begin{array}{l} a _ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}} = \gamma_ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}} - \beta_ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}} \\ = \exp \left(\log \left(\gamma_ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}}\right)\right) - \exp \left(\log \left(\beta_ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}}\right)\right) \\ = \exp \left(\log (\gamma_ {i}) + \Sigma_ {j = 1} ^ {n} \log \left(x _ {j} ^ {i _ {j}}\right)\right) - \exp \left(\log (\beta_ {i}) + \Sigma_ {j = 1} ^ {n} \log \left(x _ {j} ^ {i _ {j}}\right)\right) \\ \end{array}
$$

The argument of each exponential function is a sum of single-variable functions and constants. Without loss of generality, a set of single-variable functions can be defined such that:

$$
a _ {i} \Pi_ {j = 1} ^ {n} x _ {j} ^ {i _ {j}} = \exp \left(\Sigma_ {j = 1} ^ {n} g _ {i, j} (x _ {j})\right) - \exp \left(\Sigma_ {j = 1} ^ {n} h _ {i, j} (x _ {j})\right)
$$

Since this holds for any  $a_{i}\Pi_{j = 1}^{n}x_{j}^{i_{j}}$  and all  $i\in \mathcal{I}$ , it follows that:

$$
y (\vec {\mathbf {x}}) = \sum_ {i \in \mathcal {I}} \exp \left(\Sigma_ {j = 1} ^ {n} g _ {i, j} (x _ {j})\right) - \exp \left(\Sigma_ {j = 1} ^ {n} h _ {i, j} (x _ {j})\right)
$$

![](images/91e8d58e318bc7b1f3a34ad0ec60ce1a0fca9ac498b5402c7087d4a0ec0c83fb.jpg)

This result is fundamental to the paper. Since every continuous function can be approximated with multi-variable polynomials, it follows that every continuous function can be approximated with positive and negative exponential functions. Some consideration must be given to single-variable function approximators, since it is the main building block for the representation theorem. A lot of design and thought went into choosing a single-variable function approximator.

# 4 Single-Variable Function Approximation in Atlas

Each single-variable function in Atlas is approximated with uniform cubic B-spline basis functions. The choice was made to use uniform cubic B-splines due to their excellent performance and robustness to catastrophic forgetting. Each basis function is multiplied by a parameter and summed together. The number of basis functions is typically fixed and remains unchanged. With uniform B-splines, each basis function is scaled so that the unit interval is uniformly partitioned as in Figure 1.

Definition 1 ( $\rho$ -density B-spline function). A  $\rho$ -density B-spline function is a uniform cubic B-spline function with  $2^{\rho + 2}$  basis functions:

$$
f (x) = \sum_ {i = 1} ^ {2 ^ {\rho + 2}} \theta_ {i} S _ {i} (x) = \sum_ {i = 1} ^ {2 ^ {\rho + 2}} \theta_ {i} S (w _ {i} x + b _ {i}) = \sum_ {i = 1} ^ {2 ^ {\rho + 2}} \theta_ {i} S ((2 ^ {\rho + 2} - 3) x + 4 - i)
$$

The limitation of uniform B-spline basis functions is that increasing the number of basis functions would alter each basis function, which means any previously learned parameters would either be discarded or modified in some undetermined way. If one instead used the familiar Fourier basis functions, then one could add more basis functions, but initialise their coefficients to be zero. Recall that adding zero to a function does not change it, leaving the overall model unchanged. In order to achieve something similar with B-spline basis functions, instead of using a single density, we propose adding together uniform B-spline basis functions with different densities, as illustrated in Figure 1.

![](images/52504ecbf784a4a0b8160a2e548e163a6d3f7e361a3deba5cf81cac71db2b672.jpg)  
(a)  $\rho = 0$

![](images/8e70dc5ffc4bf644f079d8bd08ce679bd1ec3851b52a891668f376978a087bd5.jpg)  
(b)  $\rho = 1$

![](images/5fc972418905ef02c953a2c79f03613d6c25d8861ce59e92ad60ba5a2b6881b5.jpg)  
Figure 1: Doubling densities of basis functions.  
(c)  $\rho = 2$

The minimum number of cubic B-spline basis functions is four; we define this as a density  $\rho = 0$ . A density of  $\rho = 1$  is (doubled) eight uniform cubic B-spline basis functions. A density of  $\rho = 2$  is sixteen uniform cubic B-spline basis functions, and so forth. Atlas uses a double-summation, summing together all cubic B-spline basis functions, for each density up to some maximum fixed density  $r$ .

Definition 2 (mixed-density B-spline function). A mixed-density B-spline function is a single-variable function approximator that is obtained by summing together different  $\rho$ -density B-spline functions. Only the maximum  $\rho$ -density B-spline function has trainable parameters, the others are constant. Mixed-density B-spline functions are of the form:

$$
f (x) = \sum_ {\rho = 0} ^ {r} \sum_ {i = 1} ^ {2 ^ {\rho + 2}} \theta_ {\rho , i} S _ {\rho , i} (x)
$$

Analytically, we can choose all the new parameters  $\theta_{r + 1,i} = 0$ ,  $\forall i \in \mathbf{N}$  such that:

$$
f (x) = \sum_ {\rho = 0} ^ {r} \sum_ {i = 1} ^ {2 ^ {\rho + 2}} \theta_ {\rho , i} S _ {\rho , i} (x) = \sum_ {\rho = 0} ^ {r + 1} \sum_ {i = 1} ^ {2 ^ {\rho + 2}} \theta_ {\rho , i} S _ {\rho , i} (x)
$$

The last thing to note is that only the parameters for the largest specified density are trainable, in contrast to smaller density parameters that are fixed constants. It is possible to create a minimal model with  $r = 0$  initialised at zero, and train the model until convergence. Then one can create a new model with  $r = 1$ , by subsuming the previous model's parameters. The larger and more expressive model can be trained up to convergence. This process of training and expansion can be continued indefinitely and is shown in Figure 2. Every single-variable function in Atlas is a mixed-density B-spline function.

![](images/9fabaea2a5c4e9ec899bb7523624b796d0736a266c74fef45ac19e77379d9ceb.jpg)

![](images/9bb58bfb3ccef97c8a83c1f762d72e9351c7b31feb4728a9011fdb0595c2947f.jpg)

![](images/41662bc6b159c672f4d6d100bdca75fc4606249a2a93621f32de43eb1cdb8d8c.jpg)

![](images/6449206131af5da8937ad1ef308da881f1d62d135c983947c1d4abe363c381a0.jpg)  
Figure 2: Doubling densities of basis functions before and after training.

![](images/177f6514c3c3b1facf54ff200805f4ad38a8828781e5cfd51a69f0db41f58ae3.jpg)

![](images/27c0a39e4783e3c7dd6ba9297060df6dd51362719449bc08bf2661c6b9caec6e.jpg)

# 5 Atlas

Atlas is named for carrying the burden of all it must remember, after the Titan god Atlas in Greek mythology who was tasked with holding the weight of the world.

Theorem 2 (Atlas representation theorem). Any multi-variable polynomial  $y(\vec{\mathbf{x}})$  of  $n$  variables over the positive orthant, can be exactly represented by continuous single-variable functions  $f_{j}(x_{j})$ ,  $g_{i,j}(x_j)$ , and  $h_{i,j}(x_j)$  in the form:

$$
y (\vec {\mathbf {x}}) = \sum_ {j = 1} ^ {n} f _ {j} (x _ {j}) + \sum_ {k = 1} ^ {\infty} \frac {1}{k ^ {2}} \exp \left(\Sigma_ {j = 1} ^ {n} g _ {k, j} (x _ {j})\right) - \frac {1}{k ^ {2}} \exp \left(\Sigma_ {j = 1} ^ {n} h _ {k, j} (x _ {j})\right)
$$

Definition 3 (Atlas). Atlas is a function approximator of  $n$  variables, with mixed-density B-spline functions  $f_{j}(x_{j}), g_{i,j}(x_{j})$ , and  $h_{i,j}(x_j)$  in the form:

$$
A (\vec {\mathbf {x}}) := \sum_ {j = 1} ^ {n} f _ {j} (x _ {j}) + \sum_ {k = 1} ^ {M} \frac {1}{k ^ {2}} \exp \left(\Sigma_ {j = 1} ^ {n} g _ {k, j} (x _ {j})\right) - \frac {1}{k ^ {2}} \exp \left(\Sigma_ {j = 1} ^ {n} h _ {k, j} (x _ {j})\right)
$$

Atlas is equivalently given by the compact notation:

$$
A (\vec {\mathbf {x}}) := F (\vec {\mathbf {x}}) + \sum_ {k = 1} ^ {M} \frac {1}{k ^ {2}} \exp (G _ {k} (\vec {\mathbf {x}})) - \frac {1}{k ^ {2}} \exp (H _ {k} (\vec {\mathbf {x}}))
$$

The number of exponential terms can be increased without changing the output of the model. We can choose to initialise  $G_{M + 1}(\vec{\mathbf{x}}) = 0$  and  $H_{M + 1}(\vec{\mathbf{x}}) = 0$ , such that the model capacity can be increased without changing the output of the model:

$$
\begin{array}{l} A _ {M + 1} (\vec {\mathbf {x}}) = \sum_ {k = 1} ^ {M + 1} \frac {1}{k ^ {2}} \exp (G _ {k} (\vec {\mathbf {x}})) - \frac {1}{k ^ {2}} \exp (H _ {k} (\vec {\mathbf {x}})) \\ = \frac {1}{(M + 1) ^ {2}} \exp (G _ {M + 1} (\vec {\mathbf {x}})) - \frac {1}{(M + 1) ^ {2}} \exp (H _ {M + 1} (\vec {\mathbf {x}})) + A (\vec {\mathbf {x}}) \\ = \frac {1}{(M + 1) ^ {2}} \exp (0) - \frac {1}{(M + 1) ^ {2}} \exp (0) + A (\vec {\mathbf {x}}) \\ = A (\vec {\mathbf {x}}) \\ \end{array}
$$

Atlas is a universal function approximator with robust memory retention. It possesses three properties atypical of most universal function approximators:

1. The activity within Atlas is sparse - most neural units are zero and inactive.  
2. The gradient vector with respect to trainable parameters is bounded regardless of the size and capacity of the model, so training is numerically stable for many possible training hyper-parameters.  
3. Inputs that are sufficiently far from each other have orthogonal representations.

Property 1 (Sparsity). For any  $\vec{\mathbf{x}}\in D(A)\subset R^n$  and bounded trainable parameters  $\theta_{i}$  with index set  $\Theta$ , the gradient vector of trainable parameters (for Atlas) is sparse:

$$
\left\| \vec {\nabla} _ {\vec {\theta}} A (\vec {\mathbf {x}}) \right\| _ {0} = \sum_ {i \in \Theta} d _ {H a m m i n g} \left(\frac {\partial A}{\partial \theta_ {i}} (\vec {\mathbf {x}}), 0\right) \leq 4 n (2 M + 1)
$$

Remark. For a fixed number of variables  $n$ , the model has a total of  $n2^{r + 2}(2M + 1)$  trainable parameters. The gradient vector has a maximum of  $4n(2M + 1)$  non-zero entries, which is independent of  $r$ . Recall that only the maximum density  $(\rho = r)$  cubic B-spline function has trainable parameters. The fraction of trainable basis functions that are active is at most  $2^{-r}$ . Sparsity entails efficient implementation, and suggests possible memory retention and robustness to catastrophic forgetting.

Property 2 (Gradient flow attenuation). For any  $\vec{\mathbf{x}} \in D(A) \subset R^n$  and bounded trainable parameters  $\theta_i$  with index set  $\Theta$ : if all the mixed-density  $B$ -spline functions are bounded, then the gradient vector of trainable parameters for Atlas is bounded:

$$
\left\| \vec {\nabla} _ {\vec {\theta}} A (\vec {\mathbf {x}}) \right\| _ {1} = \sum_ {\theta_ {i} \in \Theta} \left| \frac {\partial A}{\partial \theta_ {i}} (\vec {\mathbf {x}}) \right| <   U
$$

Remark. For a fixed number of variables  $n$ , the model has a total of  $n2^{r + 2}(2M + 1)$  trainable parameters. The factor of  $k^{-2}$  inside the expression for Atlas is necessary to ensure the sum is convergent in the limit of infinitely many exponential terms  $M \to \infty$ . Only the maximum density  $(\rho = r)$  cubic B-spline function has trainable parameters, so that the gradient vector is bounded in the limit of arbitrarily large densities  $r \to \infty$ . It is worth recalling that at most four basis functions are active for uniform cubic B-spline functions, regardless of the density, but the smaller densities cannot be trainable, otherwise this property does not hold. The gradient vector has bounded L1 norm for any number of basis functions and exponential terms. The bounded gradient vector implies that Atlas is numerically stable during training, regardless of its size or parameter count.

Property 3 (Distal orthogonality). For any  $\vec{\mathbf{x}},\vec{\mathbf{y}}\in D(A)\subset R^n$  and bounded trainable parameters  $\theta_{i}$ , there exists a  $\delta >0$  such that:

$$
\min  _ {j = 1, \dots , n} \left\{\left| x _ {j} - y _ {j} \right| \right\} > \delta \Rightarrow \langle \vec {\nabla} _ {\vec {\theta}} A (\vec {\mathbf {x}}), \vec {\nabla} _ {\vec {\theta}} A (\vec {\mathbf {y}}) \rangle = 0
$$

Remark. Two points that sufficiently differ in each input variable have orthogonal parameter gradients. It is worth mentioning that the condition resembles a cross-like region in two variables, and planes that intersect in higher dimensions. Distal orthogonality means Atlas is reasonably robust to catastrophic forgetting.

The absolutely convergent series of scale factors  $k^{-2}$  was chosen for numerical stability and to ensure the model is absolutely convergent. Another feature is that the series of scale factors also breaks the symmetry that would otherwise exist if all mixed-density B-spline functions were initialised to zero. Initialising all the parameters to be zero is a departure from the conventional approach of random initialisation.

Atlas can be implemented with 1D convolutional layers and dense layers, as outlined in Figure 3. Only some of the dense layers have trainable parameters. The same basis functions are used for each input variable, hence the choice of 1D convolutions with fixed parameters and spline activations to correctly implement mixed-density B-spline functions. Dense core corresponds to the lower  $\rho$ -density B-spline function parameters that are not trainable. Dense summation sums together the positive and negative exponential functions with the appropriate scaling factors. It is worth mentioning that an efficient implementation of Atlas converged on using 1D convolutional layers as a method for approximating high-dimensional functions, which potentially suggests a deeper connection to other high-dimensional convolutional models (with small filters).

![](images/e4eed5b78d3eb58cf7635cabb20c6af9ff1d6e768db17688f3ab98f7df78ac6d.jpg)  
Figure 3: Diagram of an implementation of Atlas using 1D convolution layers and dense layers.

# 6 Methodology

Multiple experiments were conducted to test the function approximation capability and memory retention properties of Atlas. For the sake of brevity, only the results of Experiment A are presented in this paper. Additional experiments can be found in the supplementary material.

Experiment A was constructed for two reasons: to showcase model training-expansion cycles for a two-variable function; and to demonstrate Atlas memory retention. The swiss-roll target function was chosen because it can be visually analysed, and it is a well-known low-dimensional problem [12], not trivially fit with a standard ANN.

Experimentation was performed with Python and TensorFlow. Experiment A was performed on a personal laptop with a 7th generation i7 Intel processor and took a few hours to finish thirty trials. The loss function chosen for training and evaluation is the mean absolute error (MAE). The training data set and test set in all experiments had 10000 data points, sampled uniformly at random. Gaussian noise with standard deviation 0.1 was added to all training and test data target values. The test set was also used as a validation set to quantify the test error during training. All models were trained with a learning rate of 0.01 with the Adam optimizer. All models and experiments used batch sizes of 100 during training.

To test memory retention, two tasks, presented to Atlas one after the other, were constructed. The details of each task are given below.

Task 1 The training and test sets were sampled uniformly from the Task 1 target function over the domain  $[0., 1.]^2$ , with Gaussian noise added to the target values. The initial Atlas model was instantiated as a two-variable function that maps to a one-dimensional output, with  $r = 0$  and  $M = 0$  such that it is a minimally expressive model. The model was evaluated and trained for 30 epochs. After training the Atlas ANN model was expanded using the built-in methods, such that  $r$  increases

with one, and  $M$  is increased by two:  $r' = r + 1$  and  $M' = M + 2$ . This training-expansion process is repeated four times. The output of the model is presented at the end of each expansion iteration. Where the radius is measured from the centre of the domain [0.5, 0.5], such that the Task 1 target function is given by:

$$
\begin{array}{l} r = \sqrt {(x _ {1} - \frac {1}{2}) ^ {2} + (x _ {2} - \frac {1}{2}) ^ {2}} \\ \theta = \tan^ {- 1} \left(\left(x - \frac {1}{2}\right) ^ {2}, \left(y - \frac {1}{2}\right) ^ {2}\right) \\ Y _ {A} = Y _ {A} \left(x _ {1}, x _ {2}\right) = \sin (3 0 r + \theta) + 2 \\ \end{array}
$$

Task 2 The test sets were sampled uniformly from the Task 2 target function over the domain  $[0.,1.]^2$ , with Gaussian noise added to the target values. The training sets were sampled uniformly over the domain  $[0.45,0.55]^2$ , and target values of zero with added Gaussian noise. All models were trained for 6 epochs. The Task 2 target function is given by:

$$
Y _ {A} ^ {\prime} (x _ {1}, x _ {2}) = \left\{ \begin{array}{l l} 0 & 0. 4 5 <   x _ {1} <   0. 5 5, \text {a n d} 0. 4 5 <   x _ {2} <   0. 5 5 \\ Y _ {A} (x _ {1}, x _ {2}) & \text {o t h e r w i s e .} \end{array} \right.
$$

Task 2 effectively tests if a model changes only where new data was presented, with off-target effects leading to larger test MAE.

# 7 Results

Task 1 The output of the model after training for each expansion iteration is presented in Figure 4. The newly created trainable parameters for each expansion is set to zero, such that the expanded models are equal to the previous model before training, similar to Figure 2.

![](images/8d2d5da5e18c8e5cc16bfbb24a482e0cbfc41d887b84291b3eab223c8ae129b1.jpg)  
Figure 4: Outputs of the model during successive training and expansion iterations.

![](images/7575c239774dae59d3bf286c7782a941cde5cb1046aeafe89b1f39d6a3c36352.jpg)

![](images/285d8f143305bdbd1edc2eccdc70b84eb499833c58dae50b8241bf6574a5d42a.jpg)

![](images/db49bf3c7fd22c25cc1e5a6330003bd92764f32dd5004e8e52aa95966ca51078.jpg)  
Furreee

The mean training and validation loss curves are presented in Figure 5, with standard deviation shaded in. There is markedly little variation. The initial model converged almost immediately, and practically resembles a constant function as seen in Figure 4. Each expansion iteration is indicated with a vertical line. It is clear that the model was able to better approximate the target function with increased capacity.

Task 2 The final model from Task 1, with  $r = 4$  and  $M = 8$ , was trained on the subset  $[0.45, 0.55]^2$ . The outputs of the model and the target functions were visualised in Figure 6 with grid-sampled points. The images from the left: Task 1 target function; model output after fully training on Task 1; Task 2 target function; model output after training on Task 2; The absolute difference between the models' first and second output. A model with nearly perfect memory retention would only differ in the small square region  $[0.45, 0.55] \times [0.45, 0.55]$  in the centre of the unit square.

It is evident from Figure 6 that there are some off-target effects, and catastrophic forgetting. However, as predicted there are regions that were left untouched. Only a small cross-shaped region was susceptible to off-target effects and catastrophic forgetting.

![](images/a6a7a32c9adde93e97674ed69c37c7e0d880d8705f75cfe517136fc2522e4f28.jpg)  
Figure 5: Training and validation MAE during the course of training on Task 1

![](images/c30efebc8f0c2587424432e496f320edfc9120fc35bb5b56c771dedaad7c1532.jpg)  
Figure 6: Visual inspection of target functions and model outputs over Task 1 and Task 2.

# 8 Conclusion

The main contribution of the paper is theoretical and technical. A representation theorem is presented that outlines how to approximate multi-variable functions with single-variable functions, and exponential functions. Atlas approximates all arbitrary single-variable functions with mixed-density B-spline functions. Atlas is constructed in such a way that the gradient vector with respect to trainable parameters is bounded, regardless of how large an Atlas model is. Atlas was shown to exhibit intrinsic memory retention, except for a small cross-shaped region of interference and off-target effects.

The chosen experiment demonstrated the expansion abilities of Atlas, and its memory retention which is interesting. The specific target function is well-studied. It was mainly chosen for demonstration purposes. Higher-dimensional tests are needed to evaluate practical utility of Atlas. Since many of the basis functions are zero, the presented implementation of Atlas does redundant calculations. Improving efficiency is an ongoing research goal.

As far as societal impacts are concerned: It is possible that Atlas could allow for the creation of more powerful machine learning algorithms, that require less resources to train and deploy. Further testing is needed to make any concrete claim.

# References

[1] K. Branson. A practical review of uniform b-splines, 2004.  
[2] A. S. Douzette. B-splines in machine learning. Master's thesis, Department of Mathematics, University of Oslo, 2017.  
[3] R. M. French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128-135, 1999.  
[4] R. Hadsell, D. Rao, A. A. Rusu, and R. Pascanu. Embracing change: Continual learning in deep neural networks. Trends in cognitive sciences, 24(12):1028-1040, 2020.  
[5] B. Hanin. Which neural net architectures give rise to exploding and vanishing gradients? Advances in neural information processing systems, 31, 2018.

[6] B. Hanin. Universal function approximation by deep neural nets with bounded width and relu activations. Mathematics, 7(10):992, 2019.  
[7] K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989.  
[8] P. Kaushik, A. Gain, A. Kortylewski, and A. Yuille. Understanding catastrophic forgetting and remembering in continual learning with optimal relevance mapping. CoRR, abs/2102.11343, 2021. URL https://arxiv.org/abs/2102.11343.  
[9] R. Kemker, M. McClure, A. Abitino, T. Hayes, and C. Kanan. Measuring catastrophic forgetting in neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
[10] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521-3526, 2017. ISSN 0027-8424. doi: 10.1073/pnas.1611835114. URL https://www.pnas.org/content/114/13/3521.  
[11] A. Kratsios. The universal approximation property: Characterization, construction, representation, and existence. Annals of Mathematics and Artificial Intelligence, 89(5):435-469, 2021. doi: 10.1007/s10472-020-09723-1.  
[12] S. H. Lane, M. Flax, D. Handelman, and J. Gelfand. Multi-layer perceptrons with b-spline receptive field functions. In Advances in Neural Information Processing Systems, pages 684-692, 1991.  
[13] M. McCloskey and N. J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In *Psychology of learning and motivation*, volume 24, pages 109–165. Elsevier, 1989.  
[14] K. McRae and P. A. Hetherington. Catastrophic interference is eliminated in pretrained networks. In Proceedings of the 15th Annual Conference of the Cognitive Science Society, pages 723-728, 1993.  
[15] A. Robins. Catastrophic forgetting, rehearsal and pseudorehearsal. _Connection Science_, 7(2): 123-146, 1995.  
[16] S. Scardapane, M. Scarpiniti, D. Comminiello, and A. Uncini. Learning activation functions from data using cubic spline interpolation. In *Italian Workshop on Neural Nets*, pages 73–83. Springer, 2017.  
[17] Y. Shin and J. Ghosh. The pi-sigma network: an efficient higher-order neural network for pattern classification and function approximation. In IJCNN-91-Seattle International Joint Conference on Neural Networks, volume i, pages 13-18 vol.1, 1991. doi: 10.1109/IJCNN.1991.155142.  
[18] H. van Deventer, P. J. van Rensburg, and A. Bosman. Kasam: Spline additive models for function approximation, 2022. URL https://arxiv.org/abs/2205.06376.
