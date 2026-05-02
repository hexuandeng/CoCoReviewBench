# LEARNING ONE-HIDDEN-LAYER NEURAL NETWORKS WITH LANDSCAPE DESIGN

Anonymous authors

Paper under double-blind review

# ABSTRACT

We consider the problem of learning a one-hidden-layer neural network: we assume the input  $x \in \mathbb{R}^d$  is from Gaussian distribution and the label  $y = a^\top \sigma(Bx) + \xi$ , where  $a$  is a nonnegative vector in  $\mathbb{R}^m$  with  $m \leq d$ ,  $B \in \mathbb{R}^{m \times d}$  is a full-rank weight matrix, and  $\xi$  is a noise vector. We first give an analytic formula for the population risk of the standard squared loss and demonstrate that it implicitly attempts to decompose a sequence of low-rank tensors simultaneously. Inspired by the formula, we design a non-convex objective function  $G(\cdot)$  whose landscape is guaranteed to have the following properties:

1. All local minima of  $G$  are also global minima.  
2. All global minima of  $G$  correspond to the ground truth parameters.  
3. The value and gradient of  $G$  can be estimated using samples.

With these properties, stochastic gradient descent on  $G$  provably converges to the global minimum and learn the ground-truth parameters. We also prove finite sample complexity results and validate the results by simulations.

# 1 INTRODUCTION

Scalable optimization has been playing crucial roles in the success of deep learning, which has immense applications in artificial intelligence. Remarkably, optimization issues are often addressed through designing new models that make the resulting training objective functions easier to be optimized. For example, over-parameterization Livni et al. (2014), batch-normalization Ioffe & Szegedy (2015), and residual networks He et al. (2016a;b) are often considered as ways to improve the optimization landscape of the resulting objective functions.

How do we design models and objective functions that allow efficient optimization with guarantees? Towards understanding this question in a principled way, this paper studies learning neural networks with one hidden layer. Roughly speaking, we will show that when the input is from Gaussian distribution and under certain simplifying assumptions on the weights, we can design an objective function  $G(\cdot)$ , such that

[a] all local minima of  $G(\cdot)$  are global minima

[b] all the global minima are the desired solutions, namely, the ground-truth parameters (up to permutation and some fixed transformation).

We note that designing such objective functions is challenging because 1) the natural  $\ell_2$  loss objective does have bad local minimum, and 2) due to the permutation invariance<sup>1</sup>, the objective function inherently has to contain an exponential number of isolated local minima.

# 1.1 SETUP AND KNOWN ISSUES WITH PROPER LEARNING

We aim to learn a neural network with a one-hidden-layer using a non-convex objective function. We assume input  $x$  comes from Gaussian distribution and the label  $y$  comes from the model

$$
y = a ^ {\star} ^ {\top} \sigma \left(B ^ {\star} x\right) + \xi \tag {1.1}
$$

where  $a^{\star} \in \mathbb{R}^{d}, B^{\star} \sim \mathbb{R}^{m \times d}$  are the ground-truth parameters,  $\sigma(\cdot)$  is a element-wise non-linear function, and  $\xi$  is a noise vector with zero mean. Here we can without loss of generality assume  $x$  comes from spherical Gaussian distribution  $\mathcal{N}(0, \mathrm{Id}_{d \times d})$ .<sup>2</sup>

For technical reasons, we will further assume  $m \leq d$  and that  $a^{\star}$  has non-negative entries.

The most natural learning objective is perhaps the  $\ell_2$  loss function, given the additive noise. Concretely, we can parameterize with training parameters  $a\in \mathbb{R}^d$ ,  $B\sim \mathbb{R}^{m\times d}$  of the same dimension as  $a^\star$  and  $B^{\star}$  correspondingly,

$$
\hat {y} = a ^ {\top} \sigma (B x), \tag {1.2}
$$

and then use stochastic gradient descent to optimize the  $\ell_2$  loss function. When we have enough training examples, we are effectively minimizing the following population risk with stochastic updates,

$$
f (a, B) = \mathbb {E} \left[ \| \hat {y} - y \| ^ {2} \right]. \tag {1.3}
$$

However, empirically stochastic gradient descent cannot converge to the ground-truth parameters in the synthetic setting above when  $\sigma(x) = \mathrm{ReLU}(x) = \max\{x, 0\}$ , even if we have access to an infinite number of samples, and  $B^{\star}$  is a orthogonal matrix. Such empirical results have been reported in Livni et al. (2014) previously, and we also provide our version in Figure 1 of Section 4. This is consistent with observations and theory that over-parameterization is crucial for training neural networks successfully Livni et al. (2014); Hardt et al. (2016); Soudry & Carmon (2016).

These empirical findings suggest that the population risk  $f(a, B)$  has spurious local minima with inferior error compared to that of the global minimum. This phenomenon occurs even if we assume we know  $a^{\star}$  or  $a^{\star} = 1$  is merely just the all one's vector. Empirically, such landscape issues seem to be alleviated by over-parameterization. By contrast, our method described in the next section does not require over-parameterization and might be suitable for applications that demand the recovery of the true parameters.

# 1.2 OUR CONTRIBUTIONS

Towards learning with the same number of training parameters as the ground-truth model, we first study the landscape of the population risk  $f(\cdot)$  and give an analytic formula for it — as an explicit function of the ground-truth parameters and training parameters with the randomness of the data being marginalized out. The formula in equation (2.3) shows that  $f(\cdot)$  is implicitly attempting to solve simultaneously a finite number of low-rank tensor decomposition problems with commonly shared components.

Inspired by the formula, we design a new training model whose associated loss function — named  $f'$  and formally defined in equation (2.5) — corresponds to the loss function for decomposing a matrix (2-nd order tensor) and a 4-th order tensor (Theorem 2.2). Empirically, stochastic gradient descent on  $f'$  learns the network as shown Section 4.

Despite the empirical success of  $f'$ , we still lack a provable guarantee on the landscape of  $f'$ . The second contribution of the paper is to design a more sophisticated objective  $G(\cdot)$  whose landscape is provably nice — all the local minima of  $G(\cdot)$  are proven to be global, and they correspond to the permutation of the true parameters. See Theorem 2.3.

Moreover, the value and the gradient of  $G$  can be estimated using samples, and there are no constraints in the optimization. These allow us to use straightforward SGD (see guarantees in Ge et al. (2015); Jin et al. (2017)) to optimize  $G(\cdot)$  and converge to a local minimum, which is also a global minimum (Corollary 2.4).

Finally, we also prove a finite-sample complexity result. We will show that with a polynomial number of samples, the empirical version of  $G$  share almost the same landscape properties as  $G$  itself (Theorem 2.7). Therefore, we can also use an empirical version of  $G$  as a surrogate in the optimization.

# 1.3 RELATED WORK

The work of Arora et al. (2014) is one of the early results on provable algorithms for learning deep neural networks, where the authors give an algorithm for learning deep generative models with sparse weights. Livni et al. (2014), Zhang et al. (2016; 2017b), and Daniely et al. (2016) study the learnability of special cases of neural networks using ideas from kernel methods. Janzamin et al. (2015) give a polynomial-time algorithm for learning one-hidden-layer neural networks with twice-differential activation function and known input distributions, using the ideas from tensor decompositions.

A series of recent papers study the theoretical properties of non-convex optimization algorithms for one-hidden-layer neural networks. Brutzkus & Globerson (2017) and Tian (2017) analyze the landscape of the population risk for one-hidden-layer neural networks with Gaussian inputs under the assumption that the weights vector associated to each hidden variable (that is, the filters) have disjoint supports. Li & Yuan (2017) proves that stochastic gradient descent recovers the ground-truth parameters when the parameters are known to be close to the identity matrix. Zhang et al. (2017a) study the optimization landscape of learning one-hidden-layer neural networks with a specific activation function, and they design a specific objective function that can recover a single column of the weight matrix. Zhong et al. (2017) study the convergence of non-convex optimization from a good initializer that is produced by tensor methods. Our algorithm works for a large family of activation functions (including ReLU) and any full-rank weight matrix. To our best knowledge, we give the first global convergence result for gradient-based methods for our general setting.

The optimization landscape properties have also been investigated on simplified neural networks models. Kawaguchi (2016) shows that the landscape of deep neural nets does not have bad local minima but has degenerate saddle points. Hardt & Ma (2017) show that re-parametrization using identity connection as in residual networks He et al. (2016a) can remove the degenerate saddle points in the optimization landscape of deep linear residual networks. Soudry & Carmon (2016) show that an over-parameterized neural network does not have bad differentiable local minimum. Hardt et al. (2016) analyze the power of over-parameterization in a linear recurrent network (which is equivalent to a linear dynamical system.)

The optimization landscape has also been analyzed for other machine learning problems, including SVD/PCA phase retrieval/synchronization, orthogonal tensor decomposition, dictionary learning, matrix completion, matrix sensing Baldi & Hornik (1989); Srebro & Jaakkola (2013); Ge et al. (2015); Sun et al. (2015); Bandeira et al. (2016); Ge et al. (2016); Bhojanapalli et al. (2016); Ge et al. (2017). Our analysis techniques build upon that for tensor decomposition in Ge et al. (2015) — we add two additional regularization terms to deal with spurious local minimum caused by the weights  $\alpha^{\star}$  and to remove the constraints.

Notations: We use  $\| \cdot \|$  to denote the Euclidean norm of a vector and spectral norm of a matrix. We use  $\| \cdot \| _F$  to denote the Frobenius/Euclidean norm of a matrix or high-order tensor. For a vector  $x$ , let  $\| x\| _0$  denotes its infinity norm and for a matrix  $A$ , let  $|A|_0$  be a shorthand for  $\| \mathrm{vec}(A)\| _0$  where  $\mathrm{vec}(A)$  is the vectorization of  $A$ .

We use  $A \otimes B$  to denote the Kronecker product of  $A$  and  $B$ , and  $A^{\otimes k}$  is a shorthand for  $A \otimes \dots \otimes A$  where  $A$  appears  $k$  times. For vectors  $a \otimes b$  and  $a^{\otimes k}$  denote the tensor product. We denote the identity matrix in dimension  $d \times d$  by  $\operatorname{Id}_{d \times d}$ , or  $\operatorname{Id}$  when the dimension is clear from the context. We will define other notations when we first use them.

# 2 MAIN RESULTS

# 2.1 CONNECTING  $\ell_2$  POPULATION RISK WITH TENSOR DECOMPOSITION

We first show that a natural  $\ell_2$  loss for the one-hidden-layer neural network can be interpreted as simultaneously decomposing tensors of different orders.

A straightforward approach of learning the model (1.1) is to parameterize the prediction by

$$
\hat {y} = a ^ {\top} \sigma (B x), \tag {2.1}
$$

where  $a \in \mathbb{R}^d$ ,  $B \sim \mathbb{R}^{m \times d}$  are the training parameters. Naturally, we can use  $\ell_2$  as the empirical loss, which means the population risk is

$$
f (a, B) = \mathbb {E} \left[ \| \hat {y} - y \| ^ {2} \right]. \tag {2.2}
$$

Throughout the paper, we use  $b_1^{\star \top}, \ldots, b_m^{\star \top}$  to denote the row vectors of  $B^{\star}$  and similarly for  $B$ . That is, we have  $B = \begin{bmatrix} b_1^{\top} \\ \vdots \\ b_m^{\top} \end{bmatrix}$  and  $B^{\star} = \begin{bmatrix} b_1^{\star \top} \\ \vdots \\ b_m^{\star \top} \end{bmatrix}$ . Let  $a_i$  and  $a_i^{\star}$ 's be the coordinates of  $a$  and  $a^{\star}$  respectively.

We give the following analytic formula for the population risk defined above.

Theorem 2.1. Assume vectors  $b_{i}, b_{i}^{\star}$ 's are unit vectors. Then, the population risk  $f$  defined in equation (2.2) satisfies that

$$
f (a, B) = \sum_ {k \in \mathbb {N}} \hat {\sigma} _ {k} ^ {2} \left\| \sum_ {i \in [ m ]} a _ {i} ^ {\star} b _ {i} ^ {\star \otimes k} - \sum_ {i \in [ m ]} a _ {i} b _ {i} ^ {\otimes k} \right\| _ {F} ^ {2} + \text {c o n s t}. \tag {2.3}
$$

where  $\hat{\sigma}_k$  is the  $k$ -th Hermite coefficient of the function  $\sigma$ . See section A.1 for a short introduction of Hermite polynomial basis.

Connection to tensor decomposition: We see from equation (2.3) that the population risk of  $f$  is essentially an average of infinite number of loss functions for tensor decomposition. For a fixed  $k \in \mathbb{N}$ , we have that the  $k$ -th summand in equation (2.3) is equal to (up to the scaling factor  $\hat{\sigma}_k^2$ )

$$
f _ {k} \triangleq \| T _ {k} - \sum_ {i \in [ m ]} a _ {i} b _ {i} ^ {\otimes k} \| _ {F} ^ {2}. \tag {2.4}
$$

where  $T_{k} = \sum_{i\in [m]}a_{i}^{\star}b_{i}^{\star \otimes k}$  is a  $k$ -th order tensor in  $(\mathbb{R}^d)^{\otimes k}$ . We note that the objective  $f_{k}$  naturally attempts to decompose the  $k$ -order rank-  $m$  tensor  $T_{k}$  into  $m$  rank-1 components  $a_1b_i^{\otimes k},\ldots ,a_m b_m^{\otimes k}$ .

The proof of Theorem 2.1 follows from using techniques in Hermite Fourier analysis, which is deferred to Section A.2.

Issues with optimizing  $f$ : It turns out that optimizing the population risk using stochastic gradient descent is empirically difficult. Figure 1 shows that in a synthetic setting where the noise is zero, the test error empirically doesn't converge to zero for sufficiently long time with various learning rate schemes, even if we are using fresh samples in iteration. This suggests that the landscape of the population risk has some spurious local minimum that is not a global minimum. See Section 4 for more details on the experiment setup.

An empirical fix.: Inspired by the connection to tensor decomposition objective described earlier in the subsection, we can design a new objective function that takes exactly the same form as the tensor decomposition objective function  $f_{2} + f_{4}$ . Concretely, let's define  $\hat{y}' = a^\top \gamma(Bx)$  where  $\gamma = \hat{\sigma}_2 h_2 + \hat{\sigma}_4 h_4$  and  $h_2(t) = \frac{1}{\sqrt{2}} (t^2 - 1)$  and  $h_4(t) = \frac{1}{\sqrt{24}} (t^4 - 6t^2 + 3)$  are the 2nd and 4th normalized probabilists' Hermite polynomials Wikipedia (2017a). We abuse the notation slightly by using the same notation to denote the its element-wise application on a vector. Now for each example we use  $\| \hat{y}' - y \|^2$  as loss function. The corresponding population risk is

$$
f ^ {\prime} (a, B) = \mathbb {E} \left[ \| \hat {y} ^ {\prime} - y \| ^ {2} \right]. \tag {2.5}
$$

Now by an extension of Theorem 2.1, we have that the new population risk is equal to the  $\hat{\sigma}_2^2 f_2 + \hat{\sigma}_4^2 f_4$ .

Theorem 2.2. Let  $f'$  be defined as in equation (2.5) and  $f_2$  and  $f_4$  be defined in equation (2.4). Assume  $b_i, b_i^*$ 's are unit vectors. Then, we have

$$
f ^ {\prime} = \hat {\sigma} _ {2} ^ {2} f _ {2} + \hat {\sigma} _ {4} ^ {2} f _ {4} + \text {c o n s t} \tag {2.6}
$$

It turns out stochastic gradient descent on the objective  $f^{\prime}(a, B)$  (with projection to the set of matrices  $B$  with row norm 1) converges empirically to the ground truth  $(a^{\star}, B^{\star})$  or one of its equivalent permutations. (See Figure 2.) However, we don't know of any existing work for analyzing the landscape of the objective  $f^{\prime}$  (or  $f_{k}$  for any  $k \geq 3$ ). We conjecture that the landscape of  $f^{\prime}$  doesn't have any spurious local minimum under certain mild assumptions on  $(a^{\star}, B^{\star})$ . Despite recent attempts on other loss functions for tensor decomposition Ge & Ma (2017), we believe that analyzing  $f^{\prime}$  is technically challenging and its resolution will be potentially enlightening for the understanding landscape of loss function with permutation invariance. See Section 4 for more experimental results.

# 2.2 LANDSCAPE DESIGN FOR ORTHOGONAL  $B^{\star}$

The population risk defined in equation (2.5) — though works empirically for randomly generated ground-truth  $(a^{\star}, B^{\star})$  — doesn't have any theoretical guarantees. It's also possible that when  $(a^{\star}, B^{\star})$  are chosen adversarially or from a different distribution, SGD no longer converges to the true parameters.

To solve this problem, we design another objective function  $G(\cdot)$ , such that the optimizer of  $G(\cdot)$  still corresponds to the ground-truth, and  $G()$  has provably nice landscape — all local minima of  $G()$  are global minima.

In this subsection, for simplicity, we work with the case when  $B^{\star}$  is an orthogonal matrix and state our main result. The discussion of the general case is deferred to the end of this Section and Section C.

We define our objective function  $G(B)$  as

$$
G (B) \triangleq \operatorname {s i g n} (\hat {\sigma} _ {4}) \mathbb {E} \left[ y \cdot \sum_ {j, k \in [ d ], j \neq k} \phi \left(b _ {j}, b _ {k}, x\right) \right] - \mu \operatorname {s i g n} (\hat {\sigma} _ {4}) \mathbb {E} \left[ y \cdot \sum_ {j \in [ d ]} \varphi \left(b _ {j}, x\right) \right] + \lambda \sum_ {i = 1} ^ {m} \left(\left\| b _ {i} \right\| ^ {2} - 1\right) ^ {2} \tag {2.7}
$$

where  $\varphi (\cdot ,\cdot)$  is defined as

$$
\varphi (v, x) = \frac {1}{8} \| v \| ^ {4} - \frac {1}{4} (v ^ {\top} x) ^ {2} \| v \| ^ {2} + \frac {1}{2 4} (v ^ {\top} x) ^ {4}. \tag {2.8}
$$

and  $\phi (\cdot ,\cdot ,\cdot)$  is defined as

$$
\begin{array}{l} \phi (v, w, x) = \frac {1}{2} \| v \| ^ {2} \| w \| ^ {2} + \langle v, w \rangle^ {2} - \frac {1}{2} \| w \| ^ {2} (v ^ {\top} x) ^ {2} - \frac {1}{2} \| v \| ^ {2} (w ^ {\top} x) ^ {2} \\ + 2 \left(v ^ {\top} x\right) \left(w ^ {\top} x\right) v ^ {\top} w + \frac {1}{2} \left(v ^ {\top} x\right) ^ {2} \left(w ^ {\top} x\right) ^ {2}. \tag {2.9} \\ \end{array}
$$

The rationale behind the choices of  $\phi$  and  $\varphi$  will only be clearer and relevant in later sections. For now, the only relevant property of them is that both are smooth functions whose derivatives are easily computable.

We remark that we can sample  $G(\cdot)$  using the samples straightforwardly — it's defined as an average of functions of examples and the parameters. We also note that only parameter  $B$  appears in the loss function. We will infer the value of  $a^{\star}$  using straightforward linear regression after we get the (approximately) accurate value of  $B^{\star}$ .

Due to technical reasons, our method only works for the case when  $a_i^\star > 0$  for every  $i$ . We will assume this throughout the rest of the paper. The general case is left for future work. Let  $a_{\max}^{\star} = \max a_i^\star$ ,  $a_{\min}^{\star} = \min a_i^\star$ , and  $\kappa^{\star} = \max a_i^{\star} / \min a_i^{\star}$ . Our result will depend on the value of  $\kappa^{\star}$ . Essentially we treat  $\kappa^{\star}$  as an absolute constant that doesn't scale in dimension. The following theorem characterizes the properties of the landscape of  $G(\cdot)$ .

Theorem 2.3. Let  $c$  be a sufficiently small universal constant (e.g.  $c = 0.01$  suffices) and suppose the activation function  $\sigma$  satisfies  $\hat{\sigma}_4 \neq 0$ . Assume  $\mu \leq c / \kappa^\star$ ,  $\lambda \geq c^{-1}a_{\max}^\star$ , and  $B^\star$  is an orthogonal matrix. The function  $G(\cdot)$  defined as in equation (2.7) satisfies that

1. A matrix  $B$  is a local minimum of  $G$  if and only if  $B$  can be written as  $B = DP B^{\star}$  where  $P$  is a permutation matrix and  $D$  is a diagonal matrix with  $D_{ii} \in \{\pm 1 \pm O(\mu a_{\max}^{\star} / \lambda)\}$ . Furthermore, this means that all local minima of  $G$  are also global.  
2. Any saddle point  $B$  has a strictly negative curvature in the sense that  $\lambda_{\min}(\nabla^2 G(B)) \geq -\tau_0$  where  $\tau_0 = c\min \{\mu a_{\min}^\star / (\kappa^\star d), \lambda\}$  
3. Suppose  $B$  is an approximate local minimum in the sense that  $B$  satisfies

$$
\| \nabla G (B) \| \leq \varepsilon \text {a n d} \lambda_ {\min } (\nabla^ {2} G (B)) \geq - \tau_ {0}
$$

Then  $B$  can be written as  $B = PDB^{\star} + EB^{\star}$  where  $P$  is a permutation matrix,  $D$  is a diagonal matrix satisfying the same bound as in bullet 1, and  $|E|_{\infty} \leq O(\varepsilon / (\hat{\sigma}_4 a_{\min}^{\star}))$ .

As a direct consequence,  $B$  is  $O_d(\varepsilon)$ -close to a global minimum in Euclidean distance, where  $O_d(\cdot)$  hides polynomial dependency on  $d$  and other parameters.

The theorem above implies that we can learn  $B^{\star}$  (up to permutation of rows and sign-flip) if we take  $\lambda$  to be sufficiently large and optimize  $G(\cdot)$  using stochastic gradient descent. In this case, the diagonal matrix  $D$  in bullet 1 is sufficiently close to identity (up to sign flip) and therefore a local minimum  $B$  is close to  $B^{\star}$  up to permutation of rows and sign flip. The sign of each  $b_{i}^{\star}$  can be recovered easily after we recover  $a$  (see Lemma 2.5 below.)

SGD converges to a local minimum Ge et al. (2015) (under the additional property as established in bullet 2 above), which is also a global minimum for the function  $G(\cdot)$ . We will prove the theorem in Section B as a direct corollary of Theorem B.1. The technical bullet 2 and 3 of the theorem is to ensure that we can use SGD to converge to a local minimum as stated below. $^6$

Corollary 2.4. In the setting of Theorem 2.3, we can use stochastic gradient descent to optimize function  $G(\cdot)$  (with fresh samples at each iteration) and converge to an approximate global minimum  $B$  that is  $\varepsilon$ -close to a global minimum in time poly  $(d,1 / \varepsilon)$ .

After approximately recovering the matrix  $B^{\star}$ , we can also recover the coefficient  $a^{\star}$  easily. Note that fixing  $B$ , we can fit  $a$  using simply linear regression. For the ease of analysis, we analyze a slightly different algorithm. The lemma below is proved in Section D.

Lemma 2.5. Given a matrix  $B$  whose rows have unit norm, and are  $\delta$ -close to  $B^{\star}$  in Euclidean distance up to permutation and sign flip with  $\delta \leq 1/(2\kappa^{\star})$ . Then, we can give estimates  $a$ ,  $B'$  (using e.g., Algorithm 1) such that there exists a permutation  $P$  where  $\|a - Pa^{\star}\|_{\infty} \leq \delta a_{\max}^{\star}$  and  $B'$  is row-wise  $\delta$ -close to  $PB^{\star}$ .

The key step towards analyzing objective  $G(B)$  is the following theorem that gives an analytic formula for  $G(\cdot)$ .

Theorem 2.6. The function  $G(\cdot)$  satisfies

$$
G (B) = 2 \sqrt {6} | \hat {\sigma} _ {4} | \cdot \sum_ {i \in [ d ]} a _ {i} ^ {\star} \sum_ {j, k \in [ d ], j \neq k} \left\langle b _ {i} ^ {\star}, b _ {j} \right\rangle^ {2} \left\langle b _ {i} ^ {\star}, b _ {k} \right\rangle^ {2} - \frac {| \hat {\sigma} _ {4} | \mu}{\sqrt {6}} \sum_ {i, j \in [ d ]} a _ {i} ^ {\star} \left\langle b _ {i} ^ {\star}, b _ {j} \right\rangle^ {4} + \lambda \sum_ {i = 1} ^ {m} (\| b _ {i} \| ^ {2} - 1) ^ {2} \tag {2.10}
$$

Theorem 2.6 is proved in Section A. We will motivate our design choices with a brief overview in Section 3 and formally analyze the landscape of  $G$  in Section B (see Theorem B.1).

Finite sample complexity bounds. Extending Theorem 2.3, we can characterize the landscape of the empirical risk  $\widehat{G}$ , which implies that stochastic gradient on  $\widehat{G}$  also converges approximately to the ground-truth parameters with polynomial number of samples.

Theorem 2.7. In the setting of Theorem 2.3, suppose we use  $N$  empirical samples to approximate  $G$  and obtain empirical risk  $\widehat{G}$ . There exists a fixed polynomial  $\mathrm{poly}(d,1 / \varepsilon)$  such that if  $N\geq$  poly(d,1/ε), then with high probability the landscape of  $\widehat{G}$  has the properties to that of  $G$  in bullet 2 and 3 of Theorem 2.3.

All of the results above assume that  $B^{\star}$  is orthogonal. Since the local minimum are preserved by linear transformation of the input space, these results can be extended to the general case when  $B^{\star}$  is not orthogonal but full rank (with some additional technicality) or the case when the dimension is larger than the number of neurons ( $m < d$ ). See Section C.

# 3 OVERVIEW: LANDSCAPE DESIGN AND ANALYSIS

In this section, we present a general overview of ideas behind the design of objective function  $G(\cdot)$ . Inspired by the formula (2.3), in Section 3.1, we envision a family of possible objective functions for which we have unbiased estimators via samples. In Section 3.2, we pick a specific function that feeds our needs: a) it has no spurious local minimum; b) the global minimum corresponds to the ground-truth parameters.

# 3.1 WHICH OBJECTIVE CAN BE ESTIMATED BY SAMPLES?

Recall that in equation (2.2) of Theorem 2.1 we give an analytic formula for the straightforward population risk  $f$ . Although the population risk  $f$  doesn't perform well empirically, the lesson that we learn from it helps us design better objective functions. One of the key fact that leads to the proof of Theorem 2.1 is that for any continuous and bounded function  $\gamma$ , we have that

$$
\mathbb {E} \left[ y \cdot \gamma \left(b _ {i} ^ {\top} x\right) \right] = \sum_ {k \in \mathbb {N}} \hat {\gamma} _ {k} \hat {\sigma} _ {k} \left(\sum_ {j \in [ d ]} a _ {j} ^ {\star} \left\langle b _ {j} ^ {\star}, b _ {i} \right\rangle^ {k}\right).
$$

Here  $\hat{\sigma}_k$  and  $\hat{\gamma}_k$  are the  $k$ -th Hermite coefficient of the function  $\sigma$  and  $\gamma$ . That is, letting  $h_k$  the  $k$ -th normalized probabilists' Hermite polynomials Wikipedia (2017a) and  $\langle \cdot, \cdot \rangle$  be the standard inner product between functions, we have  $\hat{\sigma}_k = \langle h_k, \sigma \rangle$ .

Note that  $\gamma$  can be chosen arbitrarily to extract different terms. For example, by choosing  $\gamma = h_k$ , we obtain that

$$
\mathbb {E} \left[ y \cdot h _ {k} \left(b _ {i} ^ {\top} x\right) \right] = \hat {\sigma} _ {k} \sum_ {j \in [ d ]} a _ {j} ^ {\star} \left\langle b _ {j} ^ {\star}, b _ {i} \right\rangle^ {k}. \tag {3.1}
$$

That is, we can always access functions forms that involves weighted sum of the powers of  $\langle b_i^\star ,b_j\rangle$  as in RHS of equation (3.1). Using a bit more technical tools in Fourier analysis (see details in Section A), we claim that most of the symmetric polynomials over variables  $\langle b_i^\star ,b_j\rangle$  can be estimated by samples:

Claim 3.1 (informal). For any polynomial  $p(\cdot)$  over a single variable, there exists a corresponding function  $\phi^p$  such that

$$
\mathbb {E} [ y \cdot \phi^ {p} (B, x) ] = \sum_ {j} a _ {j} ^ {\star} \sum_ {i} p \left(\left\langle b _ {j} ^ {\star}, b _ {i} \right\rangle\right) \tag {3.2}
$$

Moreover, for an any polynomial  $q(\cdot, \cdot)$  over two variables, there exists corresponding  $\phi^q$  such that

$$
\mathbb {E} \left[ y \cdot \phi^ {q} (B, x) \right] = \sum_ {j} a _ {j} ^ {\star} \sum_ {i, k} q \left(\left\langle b _ {j} ^ {\star}, b _ {i} \right\rangle , \left\langle b _ {k} ^ {\star}, b _ {i} \right\rangle\right) \tag {3.3}
$$

We will not prove these two general claims. Instead, we only focus on the formulas in Theorem A.5 and Theorem A.6, which are two special cases of the claims above.

Motivated by Claim A.3, in the next subsection, we will pick an objective function which has no spurious local minimum among those functional forms on the right-hand sides of equation (3.2) and (3.3).

# 3.2 WHICH OBJECTIVE HAS NO SPURIOUS LOCAL MINIMA?

As discussed briefly in the introduction, one of the technical difficulties to design and analyze objective functions for neural networks comes from the permutation invariance — if a matrix  $B$  is a good solution, then any permutation of the rows of  $B$  still gives an equally good solution (if we

also permute the coefficients in  $a$  accordingly). We only know of a very limited number of objective functions that guarantee to enjoy permutation invariance and have no spurious local minima Ge et al. (2015). We start by considering the objective function used in Ge et al. (2015),

$$
\min P (B) = \sum_ {i} \sum_ {j \neq k} \langle b _ {i} ^ {\star}, b _ {j} \rangle^ {2} \langle b _ {i} ^ {\star}, b _ {k} \rangle^ {2}
$$

$$
s. t. \forall i \in [ d ], \| b _ {i} \| = 1 \tag {3.4}
$$

Note that here we overload the notation by using  $b_{i}^{\star}$ 's to denote a set of fixed vectors that we wanted to recover and using  $b_{i}$ 's to denote the variables. Careful readers may notice that  $P(B)$  doesn't fall into the family of functions that we described in the previous section (that is, RHS equation of (3.2) and (3.3)), because it lacks the weighting  $a_{i}^{\star}$ 's. We will fix this issue later in the subsection. Before that we first summarize the nice properties of the landscape of  $P(B)$ .

For the simplicity of the discussion, let's assume  $B^{\star}$  forms an orthonormal matrix in the rest of the subsection. Then, any permutation and sign-flip of the rows of  $B^{\star}$  leads to a global minimum of  $P(\cdot)$  — when  $B = SQB^{\star}$  with a permutation matrix  $Q$  and a sign matrix  $S$  (diagonal with  $\pm 1$ ), we have that  $P(B) = 0$  because one of  $\langle b_i^\star, b_j\rangle^2$  and  $\langle b_i^\star, b_k\rangle^2$  has to be zero for all  $i, j, k^7$ .

It turns out that these permutations/sign-flips of  $B^{\star}$  are also the only local minima $^{8}$  of function  $P(\cdot)$ . To see this, notice that  $P(B)$  is a degree-2 polynomial of  $B$ . Thus if we pick an index  $s$  and fix every row except for  $b_{s}$ , then  $P(B)$  is a quadratic function over unit vector  $b_{s}$  - reduces to an smallest eigenvector problem. Eigenvector problems are known to have no spurious local minimum. Thus the corresponding function (w.r.t  $b_{s}$ ) has no spurious local minimum. It turns out the same property still holds when we treat all the rows as variables and add the row-wise norm constraints.

However, there are two issues with using objective function  $P(B)$ . The obvious one is that it doesn't involve the coefficients  $a_{i}^{\star}$ 's and thus doesn't fall into the forms of equation (3.3). Optimistically, we would hope that for nonnegative  $a_{i}^{\star}$ 's the weighted version of  $P$  below would also enjoy the similar landscape property

$$
P ^ {\prime} (B) = \sum_ {i} a _ {i} ^ {\star} \sum_ {j \neq k} \langle b _ {i} ^ {\star}, b _ {j} \rangle^ {2} \langle b _ {i} ^ {\star}, b _ {k} \rangle^ {2}
$$

When  $a_{i}^{\star}$ 's are positive, indeed the global minimum of  $P'$  are still just all the permutations of the  $B^{\star}$ .<sup>9</sup> However, when  $\max a_{i}^{\star} > 2\min a_{i}^{\star}$ , we found that  $P'$  starts to have spurious local minima. It seems that spurious local minimum often occurs when a row of  $B$  is a linear combination of a smaller number of rows of  $B^{\star}$ . See Section F for a concrete example.

To remove such spurious local minima, we add a regularization term below that pushes each row of  $B$  to be close to one of the rows of  $B^{\star}$ ,

$$
R (B) = - \mu \sum_ {i} a _ {i} ^ {\star} \sum_ {j} \left\langle b _ {i} ^ {\star}, b _ {j} \right\rangle^ {4} \tag {3.5}
$$

We see that for each fixed  $j$ , the part in  $R(B)$  that involves  $b_{j}$  has the form  $-\mu \sum_{i} a_{i}^{\star} \langle b_{i}^{\star}, b_{j} \rangle^{4} = -\mu \langle \sum_{i} a_{i}^{\star} b_{i}^{\star \otimes 4}, b_{j}^{\otimes 4} \rangle$ . This is commonly used objective function for decomposing tensor  $\sum_{i} a_{i}^{\star} b_{i}^{\star \otimes 4}$ . It's known that for orthogonal  $b_{i}^{\star}$ 's, the only local minima are  $\pm b_{1}^{\star}, \ldots, \pm b_{d}^{\star}$  Ge et al. (2015). Therefore, intuitively  $R(B)$  pushes each of the  $b_{i}$ 's towards one of the  $b_{i}^{\star}$ 's. Choosing  $\mu$  to be small enough, it turns out that  $P'(B) + R(B)$  doesn't have any spurious local minimum as we will show in Section B.

Another issue with the choice of  $P^{\prime}(B) + R(B)$  is that we are still having a constraint minimization problem. Such row-wise norm constraints only make sense when the ground-truth  $B^{\star}$  is orthogonal and thus has unit row norm. A straightforward generalization of  $P(B)$  to non-orthogonal case

![](images/09edbf3f2a36ee9b533b18622b22a1460ca21c554877bb4d62783c35a426a872.jpg)  
Figure 1: Data are generated by a network with ReLU activation without noise. The training model uses the same architecture. Left: the estimated population risk doesn't converge to zero. Right: the parameter error using the surrogate in equation (4.1).

![](images/53a1287e2a9622b80d5b2c89d1ea359237c5aba83efb6c6f533abb48ee177f3a.jpg)

requires some special constraints that also depend on the covariance matrix  $B^{\star}B^{\star}^{\top}$ , which in turn requires a specialized procedure to estimate. Instead, we move the constraints into the objective function by considering adding another regularization term that approximately enforces the constraints.

It turns out the following regularizer suffices for the orthogonal case:  $S(B) = \lambda \sum_{i}(\| b_{i}\|^{2} - 1)^{2}$ . Moreover, we can extend this easily to the non-orthogonal case (see Section C) without estimating any statistics of  $B^{\star}$  in advance. We note that  $S(B)$  is not the Lagrangian multiplier and it does change the global minima slightly. We will take  $\lambda$  to be large enough so that  $\| b_{i}\|$  has to be close to 1. As a summary, we finally use the unconstrained objective

$$
\min  G (B) \triangleq P ^ {\prime} (B) + R (B) + S (B)
$$

Since  $R(B)$  and  $S(B)$  are degree-4 polynomials of  $B$ , the analysis of  $G(B)$  is much more delicate, and we cannot use much linear algebra as we could for  $P'(B)$ . See Section B for details.

Finally we note that a feature of this objective  $G(\cdot)$  is that it only takes  $B$  as variables. We will estimate the value of  $a^{\star}$  after we recover the value of  $B$ . (see Section D).

# 4 SIMULATION

In this section, we provide simple simulation results that verify that minimizing  $G(B)$  with SGD recovers a permutation of  $B^{\star}$ ; however, minimizing Equation (2.2) with SGD results in finding spurious local minima. Based on the formula for the population risk in Equation (2.3), we also verified empirically the conjecture that SGD would successfully recover  $B^{\star}$  using the activation functions  $\gamma(z) = \hat{\sigma}_2 h_2(z) + \hat{\sigma}_4 h_4(z)$ ,<sup>11</sup> even if the data were generated via a model with ReLU activation. (See Section 2.1 for the rationale behind such conjectures.)

For all of our experiments, we chose  $B^{\star} = \mathrm{Id}_{d\times d}$  with dimension  $d = 50$  and  $a^{\star} = 1$  for simplicity, and the data is generated from a one-hidden-layer network with ReLU activation without noise. We use stochastic gradient descent with fresh samples at each iteration, and we plot the (expected) population error (that is, the error on a fresh batch of examples).

To test whether SGD converges to a matrix  $B$  which is equivalent to  $B^{\star}$  up to permutation of rows, we use a surrogate error metric to evaluate whether  $B^{\star^{-1}}B$  is close to a permutation matrix. Given a matrix  $Q$  with row norm 1, let

$$
e (Q) = \min  \left\{1 - \min  _ {i} \max  _ {j} | Q _ {i j} |, 1 - \min  _ {j} \max  _ {i} | Q _ {i j} | \right\}. \tag {4.1}
$$

Then we have that if  $e(Q) \leq \varepsilon$  for some  $\varepsilon < 1/3$ , then it implies that  $Q$  is  $\sqrt{2\varepsilon}$ -close to a permutation matrix in infinity norm. On the other direction, we know that if  $e(Q) > \varepsilon$ , then  $Q$  is not  $\varepsilon$ -close to

![](images/c060a1d1e12eb6ad5e2c831231dd5ad38c59a2e294354bd8dd8ede84d8cd95ff.jpg)  
Figure 2: The labels are generated from a network with ReLU activation. We learn with  $\hat{\sigma}_2h_2 + \hat{\sigma}_4h_4$  activation. Left: the test loss subtracted by the theoretical global minimum value. Right: the error in parameter space measured by equation (4.1)

![](images/9860071a4b8dc8f986afb2f0d875a7c8dc673b768e20250f43c302dc01b6bd91.jpg)

![](images/c3c007d07f47f2dd82c2ab8d520d7c830f700c784ae0c6f58a806adb65bfd4a8.jpg)  
Figure 3: Learning with objective function  $G(\cdot)$ . Left: the test loss. Right: the error in parameter space measured by equation (4.1)

![](images/3e989803bfe33199b9f69678b8dfa9748788245e6799fc7f0561293f4db093cb.jpg)

any permutation matrix in infinity norm. The latter statement also holds when  $Q$  doesn't have row norm 1.

Figure 1 shows that without over-parameterization, using ReLU as an activation function, SGD doesn't converge to zero test error and the ground-truth parameters. We decreased step-size by a factor of 4 every 5000 number of iterations after the error plateaus at 10000 iterations. For the final 5000 iterations, the step-size is less than  $10^{-9}$ , so we can be confident that the non-zero objective value is not due to the variance of SGD. We see that none of the five runs of SGD converged to a global minimum.

Figure 2 shows that using  $\hat{\sigma}_2h_2 + \hat{\sigma}_4h_4$  as the activation function, SGD with projection to the set of matrices  $B$  with row norm 1 converges to the ground-truth parameters. We also plot the loss function which converges the value of a global minimum. (We subtracted the constant term in equation (2.6) so that the global minimum has loss 0.)

Figure 3 shows that using our objective function  $G(B)$ , the iterate converges to a permutation of the ground truth matrix  $B^{\star}$ . The fact that the parameter error goes up and down is not surprising, because the algorithm first gets close to a saddle point and then breaks ties and converges to a one of the global minima.

Finally we note that using the loss function  $G(\cdot)$  seems to require significantly larger batch (and sample complexity) to reduce the variance in the gradients estimation. We used batch size 262144 in the experiment for  $G(\cdot)$ . However, in contrast, for the  $\hat{\sigma}_2h_2 + \hat{\sigma}_4h_4$  we used batch size 8192 and for relu we used batch size 256.

# 5 CONCLUSION

In this paper we first give an analytic formula for the population risk of the standard  $\ell_2$  loss, which empirically may converge to a spurious local minimum. We then design a novel population loss that is guaranteed to have no spurious local minimum.

Designing objective functions with well-behaved landscape is an intriguing and potentially fruitful direction. We hope that our techniques can be useful for characterizing and designing the optimization landscape for other settings.

We conjecture that the objective  $\alpha f_{2} + \beta f_{4}$  has no spurious local minimum when  $\alpha, \beta$  are reasonable constants and the ground-truth parameters are in general position. We provided empirical evidence to support the conjecture.

Our results assume that the input distribution is Gaussian. Extending them to other input distributions is a very interesting open problem.

# REFERENCES

Sanjeev Arora, Aditya Bhaskara, Rong Ge, and Tengyu Ma. Provable bounds for learning some deep representations. In International Conference on Machine Learning, pp. 584-592, 2014.  
Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53-58, 1989.  
Afonso S Bandeira, Nicolas Boumal, and Vladislav Voroninski. On the low-rank approach for semidefinite programs arising in synchronization and community detection. arXiv preprint arXiv:1602.04426, 2016.  
Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Global optimality of local search for low rank matrix recovery. In Advances in Neural Information Processing Systems, pp. 3873-3881, 2016.  
Alon Brutzkus and Amir Globerson. Globally optimal gradient descent for a convnet with gaussian inputs. arXiv preprint arXiv:1702.07966, 2017.  
Amit Daniely, Roy Frostig, and Yoram Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity. In Advances In Neural Information Processing Systems, pp. 2253-2261, 2016.  
R. Ge and T. Ma. On the Optimization Landscape of Tensor Decompositions. ArXiv e-prints, June 2017.  
Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle pointsonline stochastic gradient for tensor decomposition. In Conference on Learning Theory, pp. 797-842, 2015.  
Rong Ge, Jason D. Lee, and Tengyu Ma. Matrix completion has no spurious local minimum. Advances in Neural Information Processing Systems (NIPS), 2016. URL http://arxiv.org/abs/1605.07272.  
Rong Ge, Chi Jin, and Yi Zheng. No spurious local minima in nonconvex low rank problems: A unified geometric analysis. arXiv preprint arXiv:1704.00708, 2017.  
Moritz Hardt and Tengyu Ma. Identity matters in deep learning. In 5th International Conference on Learning Representations (ICLR 2017), 2017.  
Moritz Hardt, Tengyu Ma, and Benjamin Recht. Gradient descent learns linear dynamical systems. CoRR, abs/1609.05191, 2016. URL http://arxiv.org/abs/1609.05191.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European Conference on Computer Vision, pp. 630-645. Springer, 2016b.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456, 2015.  
<sup>12</sup>See equation (2.4) for the definition of  $f_{k}$  and Theorem 2.2 for how to access  $\alpha f_{2} + \beta f_{4}$  in the setting of one-hidden-layer neural nets.

Majid Janzamin, Hanie Sedghi, and Anima Anandkumar. Beating the perils of non-convexity: Guaranteed training of neural networks using tensor methods. arXiv preprint arXiv:1506.08473, 2015.  
Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. arXiv preprint arXiv:1703.00887, 2017.  
Kenji Kawaguchi. Deep learning without poor local minima. In Advances in Neural Information Processing Systems, pp. 586-594, 2016.  
Yuanzhi Li and Yang Yuan. Convergence analysis of two-layer neural networks with relu activation. arXiv preprint arXiv:1705.09886, 2017.  
Roi Livni, Shai Shalev-Shwartz, and Ohad Shamir. On the computational efficiency of training neural networks. In Advances in Neural Information Processing Systems, pp. 855-863, 2014.  
Song Mei, Yu Bai, and Andrea Montanari. The landscape of empirical risk for non-convex losses. arXiv preprint arXiv:1607.06534, 2016.  
Ryan O'Donnell. Analysis of boolean functions. Cambridge University Press, 2014.  
Daniel Soudry and Yair Carmon. No bad local minima: Data independent training error guarantees for multi-layer neural networks. arXiv preprint arXiv:1605.08361, 2016.  
Nathan Srebro and Tommi Jaakkola. Weighted low-rank approximations. In ICML, 2013.  
Gilbert W Stewart. Matrix perturbation theory. 1990.  
Ju Sun, Qing Qu, and John Wright. When are nonconvex problems not scary? arXiv preprint arXiv:1510.06096, 2015.  
Yuandong Tian. An analytical formula of population gradient for two-layered relu network and its applications in convergence and critical point analysis. arXiv preprint arXiv:1703.00560, 2017.  
Wikipedia. Hermite polynomials — wikipedia, the free encyclopedia, 2017a. URL https://en.wikipedia.org/w/index.php?title=Hermite_polynomials&oldid=796842411. [Online; accessed 1-September-2017].  
Wikipedia. Formal power series — wikipedia, the free encyclopedia, 2017b. URL https://en.wikipedia.org/w/index.php?title=Formal_power_series&oldid=797671381. [Online; accessed 20-September-2017].  
Qiuyi Zhang, Rina Panigrahy, and Sushant Sachdeva. Electron-proton dynamics in deep learning. CoRR, abs/1702.00458, 2017a. URL http://arxiv.org/abs/1702.00458.  
Yuchen Zhang, Jason D Lee, and Michael I Jordan. 11-regularized neural networks are improperly learnable in polynomial time. In International Conference on Machine Learning, pp. 993-1001, 2016.  
Yuchen Zhang, Jason Lee, Martin Wainwright, and Michael Jordan. On the learnability of fully-connected neural networks. In Artificial Intelligence and Statistics, pp. 83-91, 2017b.  
Kai Zhong, Zhao Song, Prateek Jain, Peter L Bartlett, and Inderjit S Dhillon. Recovery guarantees for one-hidden-layer neural networks. arXiv preprint arXiv:1706.03175, 2017.
