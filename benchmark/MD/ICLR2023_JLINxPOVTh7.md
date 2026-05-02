# THE ONSET OF VARIANCE-LIMITED BEHAVIOR FOR NETWORKS IN THE LAZY AND RICH REGIMES

Anonymous authors

Paper under double-blind review

# ABSTRACT

For small training set sizes  $P$ , the generalization error of wide neural networks is well-approximated by the error of an infinite width neural network (NN), either in the kernel or mean-field/feature learning regime. However, at a critical sample size  $P^*$ , the finite-width network generalization begins to worsen compared to the infinite width performance. In this work, we empirically study the transition from the infinite width behavior to this variance-limited regime as a function of sample size  $P$  and network width  $N$ . We find that finite size effects can become relevant for very small dataset sizes going as  $P^* \sim \sqrt{N}$  for polynomial regression with ReLU networks. We discuss the source of this finite size behavior based on the variance of the NN's final neural tangent kernel (NTK). We then show how this transition can be pushed to larger  $P$  by enhancing feature learning or by ensemble averaging the network. We find that the learning curve for regression with the final NTK is an accurate approximation of the NN learning curve. Using this, we provide a toy model which also exhibits  $P^* \sim \sqrt{N}$  scaling and has  $P$ -dependent benefits from feature learning.

# 1 INTRODUCTION

Deep learning systems are achieving state of the art performance on a variety of tasks (Tan & Le, 2019; Hoffmann et al., 2022). Exactly how their generalization is controlled by network architecture, training procedure, and task structure is still not fully understood. One promising direction for deep learning theory in recent years is the infinite width limit. Under a certain parameterization, infinite width networks yield a kernel method known as the neural tangent kernel (NTK) (Jacot et al., 2018; Lee et al., 2019). Kernel methods are easier to analyze, allowing for accurate prediction of the generalization performance for wide networks in this regime (Bordelon et al., 2020; Canatar et al., 2021; Bahri et al., 2021; Simon et al., 2021). Infinite width networks can also operate in the mean-field regime if network outputs are rescaled by a small parameter  $\alpha$  that enhances feature learning (Mei et al., 2018; Chizat et al., 2019; Geiger et al., 2020b; Yang & Hu, 2020).

While infinite width networks provide useful limiting cases for deep learning theory, real networks have finite-width. Analysis at finite width is more difficult since predictions are dependent on the initialization of parameters. While several works have attempted to analyze feature evolution and kernel statistics at large but finite  $N$  (Dyer & Gur-Ari, 2020; Roberts et al., 2021), the implications of such finite size effects on generalization are not completely clear. Specifically, it is unknown at what value of sample size  $P$  finite width effects become relevant, what impact this has on the learning curve, and how it is affected by feature learning.

To identify the effect of finite width  $N$  and feature learning on the deviation from infinite width learning curves, we empirically study neural networks trained across a wide range of output scales  $\alpha$ , widths  $N$ , and training set sizes  $P$  on the simple task of polynomial regression with a ReLU neural network. Concretely, our experiments show the following:

- Learning curves for polynomial regression transition exhibit significant finite width effects very early, around  $P \sim \sqrt{N}$ . Finite-width NNs at large  $\alpha$  are always outperformed by their infinite width counterparts. We show this gap is driven primarily by variance of the predictor over initializations (Geiger et al., 2020a). Following prior work (Bahri et al., 2021), we

refer to this as the variance-limited regime. We compare three distinct ensembling methods to reduce error in this regime.

- NNs trained with feature learning show improved generalization both before and after the transition to the variance limited regime. Feature learning can be enhanced through rescaling the output of the network by a small scalar  $\alpha$  or by training on a more complex task (a higher-degree polynomial). We show that alignment between the final NTK and the target function on test data improves with feature learning and sample size.  
- We demonstrate that the learning curve for the NN is well captured by the learning curve for kernel regression with the final empirical NTK,  $\mathrm{eNTK}_f$ , as has been observed in other works (Vyas et al., 2022; Geiger et al., 2020b; Atanasov et al., 2021; Wei et al., 2022).  
- Using this correspondence between the NN and the final NTK, we provide a cursory account of how fluctuations in the final NTK over random initializations are suppressed at large width  $N$  and large feature learning strength. In a toy model, we reproduce several scaling phenomena, including the  $P \sim \sqrt{N}$  transition and the improvements due to feature learning through an alignment effect.

# 1.1 RELATED WORKS

Geiger et al. (2020a) analyzed the scaling of network generalization with the number of model parameters. Since the NTK fluctuates with variance  $O(N^{-1})$  for a width  $N$  network (Dyer & Gur-Ari, 2020; Roberts et al., 2021), they find that finite width networks in the lazy regime generically perform worse than their infinite width counterparts.

The scaling laws of networks over varying  $N$  and  $P$  were also studied, both empirically and theoretically by Bahri et al. (2021). They consider two types of learning curve scalings. First, they describe resolution-limited scaling, where either training set size or width are effectively infinite and the scaling behavior of generalization error with the other quantity is studied. Second, they analyze variance-limited scaling where width or training set size are fixed to a finite value and the other parameter is taken to infinity. While that work showed for any fixed  $P$  that the learning curve converges to the infinite width curve as  $O(N^{-1})$ , these asymptotics do not predict, for fixed  $N$ , at which value of  $P$  the NN learning curve begins to deviate from the infinite width theory. This is the focus of our work.

The contrast between rich and lazy networks has been empirically studied in several prior works. Depending on the structure of the task, the lazy regime can have either worse (Fort et al., 2020) or better (Ortiz-Jimenez et al., 2021; Geiger et al., 2020b) performance than the feature learning regime. For our setting, where the signal depends on only a small number of relevant input directions, we expect representation learning to be useful, as discussed in (Ghorbani et al., 2020; Paccolat et al., 2021b). Consequently, we posit and verify that the rich network will outperform the lazy one.

Our toy model is inspired by the literature on random feature models. Analysis of generalization for two layer networks at initialization in the limit of high dimensional data have been carried out using techniques from random matrix theory (Mei & Montanari, 2022; Hu & Lu, 2020; Adlam & Pennington, 2020a; Dhifallah & Lu, 2020; Adlam & Pennington, 2020b) and statistical mechanics (Gerace et al., 2020; d'Ascoli et al., 2020; d'Ascoli et al., 2020). Several of these works have identified that when  $N$  is comparable to  $P$ , the network generalization error has a contribution from variance over initial parameters. Further, they provide a theoretical explanation of the benefit of assembling predictions of many networks trained with different initial parameters. Recently, Ba et al. (2022) studied regression with the hidden features of a two layer network after taking one step of gradient descent, finding significant improvements to the learning curve due to feature learning.

# 2 PROBLEM SETUP AND NOTATION

We consider a supervised task with a dataset  $\mathcal{D} = \{\pmb{x}^{\mu},y^{\mu}\}_{\mu = 1}^{P}$  of size  $P$ . The pairs of data points are drawn from a population distribution  $p(x,y)$ . Our experiments will focus on training networks to interpolate degree  $k$  polynomials on the sphere (full details in Appendix A). For this task, the infinite width network learning curves can be found analytically. In particular at large  $P$  the generalization error scales as  $1 / P^2$  (Bordelon et al., 2020). We take a single output feed-forward NN  $\tilde{f}_{\theta}:\mathbb{R}^{D}\to \mathbb{R}$

with hidden width  $N$  for each layer. We let  $\theta$  denote all trainable parameters of the network. Using NTK parameterization (Jacot et al., 2018), the activations for an input  $\pmb{x}$  are given by

$$
h _ {i} ^ {(\ell)} = \frac {\sigma}{\sqrt {N}} \sum_ {j = 1} ^ {N} W _ {i j} ^ {(\ell)} \varphi \left(h _ {j} ^ {(\ell - 1)}\right), \quad \ell = 2, \dots L, \quad h _ {i} ^ {(1)} = \frac {\sigma}{\sqrt {D}} \sum_ {j = 1} ^ {D} W _ {i j} ^ {(1)} x _ {j}. \tag {1}
$$

Here, the output of the network is  $\tilde{f}_{\theta} = h_{1}^{(L)}$ . We will take  $\varphi$  to be the ReLU nonlinearity. At initialization we have  $W_{ij}\sim \mathcal{N}(0,1)$ . Consequently, the scale of the output at initialization is  $O(\sigma^L)$ . We denote the scale of the output by  $\alpha = \sigma^L$ .  $\alpha$  controls the feature learning strength of a given NN. Large  $\alpha$  corresponds to a lazy network while small  $\alpha$  yields a rich network with feature movement. More details on how  $\alpha$  controls feature learning are given in Appendix C.1.

In what follows, we will denote the infinite width NTK limit of this network by  $\mathrm{NTK}_{\infty}$ . We will denote its finite width linearization by  $\mathrm{eNTK}_0(\pmb{x},\pmb{x}^{\prime}):= \sum_{\theta}\partial_{\theta}f(\pmb{x})\partial_{\theta}f(\pmb{x}^{\prime})|_{\theta = \theta_0}$ , and we will denote its linearization around its final parameters  $\theta_f$  by  $\mathrm{eNTK}_f(\pmb{x},\pmb{x}^{\prime}):= \sum_{\theta}\partial_{\theta}f(\pmb{x})\partial_{\theta}f(\pmb{x}^{\prime})|_{\theta = \theta_f}$ .

Following other authors (Chizat et al., 2019; Adlam & Pennington, 2020a), we will take the output to be  $f_{\theta}(\pmb{x}) \coloneqq \tilde{f}_{\theta}(\pmb{x}) - \tilde{f}_{\theta_0}(\pmb{x})$ . Thus, at initialization the function output is 0. We explain this choice further in Appendix A. The parameters are then trained with full-batch gradient descent on a mean squared error loss. We denote the final network function starting from initialization  $\theta_0$  on a dataset  $\mathcal{D}$  by  $f_{\theta_0,\mathcal{D}}^*(\pmb{x})$  or  $f^*$  for short. The generalization error is calculated using a held-out test set and approximates the population risk  $E_g(f) \coloneqq \left\langle (f(\pmb{x}) - y)^2 \right\rangle_{\pmb{x},y \sim p(\pmb{x},y)}$ .

# 3 EMPIRICAL RESULTS

In this section, we will study learning curves for NNs trained on polynomial regression tasks of varying degrees. We will establish the following key observations, which we will set out to theoretically explain in Section 4.

1. Both  $\mathrm{eNTK_0}$  and sufficiently lazy networks perform strictly worse than  $\mathrm{NTK}_{\infty}$ , but the ensembled predictors approach the  $\mathrm{NTK}_{\infty}$  test error.  
2. NNs in the feature learning regime of small  $\alpha$  can outperform  $\mathrm{NTK}_{\infty}$  for an intermediate range of  $P$ . Over this range, the effect of ensembling is less notable.  
3. Even richly trained finite width NNs eventually perform worse than  $\mathrm{NTK}_{\infty}$  at sufficiently large  $P$ . However, these small  $\alpha$  feature-learning networks become variance-limited at larger  $P$  than lazy networks. Once in the variance-limited regime, all networks benefit from ensembling over initializations.  
4. For all networks, the transition to the variance-limited regime begins at a  $P^*$  that scales sub-linearly with  $N$ . The scaling we find for the task of polynomial regression is close to  $P^* \sim \sqrt{N}$ .

# 3.1 FINITE WIDTH EFFECTS CAUSE THE ONSET OF A VARIANCE LIMITED REGIME

In this section, we first investigate how finite width NN learning curves differ from infinite width NTK regression. In Figure 1 we show the generalization error  $E_{g}(f_{\theta_{0},\mathcal{D}}^{*})$  for a depth 3 network with width  $N = 1000$  trained on a quadratic  $k = 2$  and quartic  $k = 4$  polynomial regression task. Additional plots for other degree polynomials are provided in Appendix E. We sweep over  $P$  to show the effect of more data on generalization, which is the main relationship we are interested in studying. For each training set size we sweep over a grid of 20 random draws of the train set and 20 random network initializations. This for 400 trained networks in total at each choice of  $P, k, N, \alpha$ . We see that a discrepancy arises at large enough  $P$  where the neural networks begin to perform worse than  $\mathrm{NTK}_{\infty}$ .

We probe the source of the discrepancy between finite width NNs and  $\mathrm{NTK}_{\infty}$  by ensemble averaging network predictions  $\bar{f}_{\mathcal{D}}(\pmb{x}) \coloneqq \langle f_{\theta_0,\mathcal{D}}^*(\pmb{x}) \rangle_{\theta_0}$  over  $E = 20$  initializations  $\theta_0$ . In Figures 1b and 1d, we calculate the error of  $\bar{f}_{\mathcal{D}}(\pmb{x})$ , each trained on the same dataset. We then plot  $E_g(\bar{f}_{\mathcal{D}})$ . This ensembled error approximates the bias in a bias-variance decomposition (Appendix B). Thus, any gap between 1 (a) and 1 (b) is driven by variance of  $f_{\theta,\mathcal{D}}$  over  $\theta$ .

![](images/4ae50327a326cc172dced15956ba30734499d041032fbe33d47f4d0aa44148df.jpg)  
(a)  $k = 2$  generalization error

![](images/da4809f743d0f735c0e1a77b8acadd585a55168fe5e9f3587df2a638e9e11489.jpg)  
(b)  $k = 2$  20-fold ensemble error

![](images/548519d7058458b3bb8b47dcc15a5993b23605f93bd524f3e21994abbf3d87ea.jpg)  
(c)  $k = 4$  generalization error

![](images/b5ed9d50c71a506cebf877fcf3caf7923a76a06c7ddad33c99f25d54b434fd4a.jpg)  
Figure 1: Generalization errors of depth  $L = 3$  neural networks across a range of  $\alpha$  values compared to  $\mathrm{NTK}_{\infty}$ . The regression for  $\mathrm{NTK}_{\infty}$  was calculated using the Neural Tangents package (Novak et al., 2020). The exact scaling of  $\mathrm{NTK}_{\infty}$  is known to go asymptotically as  $P^{-2}$  for this task. a) Lazy networks perform strictly worse than  $\mathrm{NTK}_{\infty}$  while rich networks can outperform it for an intermediate range of  $P$  before their performance is also limited. b) Ensembling 20 networks substantially improves lazy network and  $\mathrm{eNTK}_0$  generalization, as well as asymptotic rich network generalization. This indicates that at sufficiently large  $P$ , these neural networks become limited by variance due to initialization. The error bars in a) and c) denote the variance due to both both training set and initialization. The error bars in b), d) denote the variance due to the train set.  
(d)  $k = 4$  20-fold ensemble error

We sharpen these observations with phase plots of NN generalization, variance and kernel alignment over  $P, \alpha$ , as shown in Figure 2. In Figure 2a, generalization for NNs in the rich regime (small  $\alpha$ ) have lower final  $E_{g}$  than lazy networks. As the dataset grows, the fraction of  $E_{g}$  due to initialization variance (that is, the fraction removed by assembling) strictly increases (2 (b)). We will show why this effect occurs in section 3.2. Figure 2b shows that, at any fixed  $P$ , the variance is lower for small  $\alpha$ . To measure the impact of feature learning on the eNTK  $f$ , we plot its alignment with the target function, measured as  $\frac{y^{\top} K y}{|y|^{2} \operatorname{Tr} K}$  for a test set of targets  $[y]_{\mu}$  and kernel  $[K]_{\mu \nu} = \mathrm{eNTK}_{f}(x_{\mu}, x_{\nu})$ . Alignment of the kernel with the target function is known to be related to good generalization (Canatar et al., 2021). In Section 4, we revisit these effects in a simple model which relates kernel alignment and variance reduction.

In addition to initialization variance, variance over dataset  $\mathcal{D}$  contributes to the total generalization error. Following (Adlam & Pennington, 2020b), we discuss a symmetric decomposition of the variance in Appendix B, showing the contribution from dataset variance and the effects of bagging. We find that most of the variance in our experiments is due to initialization.

We show several other plots of the results of these studies in the appendix. We show the effect of bagging (Figure 7), phase plots of different degree target functions (Figures 8, 9), phase plots over  $N, \alpha$  (Figure 10) and a comparison of network predictions against the initial and final kernel regressors (Figures 11, 12).

![](images/aec192713798854ab2a4240458913947bc140c44ea2e5f0d4b51cd7678063af6.jpg)  
(a)  $k = 3$  generalization error

![](images/53c93596f1794601f57735c6e69e77008cfe1e6eb4ad62ae3386446fb7c0524a.jpg)  
(b)  $k = 3$  variance fraction

![](images/cc79a3d2d3e03ba5fb73750c43d0a529e0c5273a339e32ed3d2ab9d308e2e945.jpg)  
(c)  $k = 3$  alignment

![](images/3c28cc6f1261c485316cedde473651be2fe88adfd9638fd835d70b2d8d90e74b.jpg)  
Figure 2: Phase plots in the  $P, \alpha$  plane of a) The log generalization error  $\log_{10} E_g(f^{\star})$ , b) The fraction of generalization error removed by ensembling  $1 - E_g(\bar{f}^{\star}) / E_g(f^{\star})$ , c) Kernel-task alignment measured by  $\frac{\mathbf{y}^T K_f \mathbf{y}}{| \mathbf{y}|^2 \operatorname{Tr} K_f}$  where  $\mathbf{y}$  and  $K_f$  are evaluated on test data. We have plotted 'x' markers in a) to show the points where the NNs were trained.  
(a)  $E_g^{NN} = E_g^{NTK_f}$

![](images/d9dfd7df1fca4f9ffc572bf1cbf0b98de5030b6680d6fc19a8f5b3da610c51ae.jpg)  
Figure 3: Performing kernel regression with the final NTK reproduces the learning curves of the neural network with high fidelity. (a) Learning curves across different laziness settings  $\alpha$  in a width 1000 network. The solid black curve is the infinite width network. Solid colored curves are the neural network generalizations. Stars represent the final kernels, and lie on top of the corresponding neural network learning curves. (b) The agreement of generalizations between neural network and final kernel across different widths and feature learning strengths.  
(b)  $E_{g}^{NN} = E_{g}^{NTK_{f}}$  across  $N,\alpha$

# 3.2 FINAL NTK VARIANCE LEADS TO GENERALIZATION PLATEAU

In this section, we show how the variance over initialization can be interpreted as kernel variance in both the rich and lazy regimes. We also show how this implies a plateau for the generalization error.

To begin, we demonstrate empirically that all networks have the same generalization error as kernel regression solutions with their final eNTKs. At large  $\alpha$ , the initial and the final kernel are already close, so this follows from earlier results of Chizat et al. (2019). In the rich regime, the properties of the  $\mathrm{eNTK}_f$  have been studied in several prior works. Several have empirically demonstrated that the  $\mathrm{eNTK}_f$  is a good match to the final network predictor for a trained network (Long, 2021; Vyas et al., 2022; Wei et al., 2022) while others have given conditions under which such an effect would hold true (Atanasov et al., 2021; Bordelon & Pehlevan, 2022). We show in Figure 3 how the final network generalization error matches the generalization error of  $\mathrm{eNTK}_f$ . As a consequence, we can use  $\mathrm{eNTK}_f$  to study the observed generalization behavior.

Next, we relate the variance of the final predictor  $f_{\theta_0,\mathcal{D}}^*$  to the corresponding infinite width network  $f_{\mathcal{D}}^{\infty}$ . The finite size fluctuations of the kernel at initialization have been studied in (Dyer & Gur-Ari, 2020; Hanin & Nica, 2019; Roberts et al., 2021). The variance of the kernel elements has been shown to scale as  $1 / N$ . We perform the following bias-variance decomposition: Take  $f_{\theta_0,\mathcal{D}}$  to be

![](images/02baf48088e7c0194e4f454309c1c3d6a1e40ac57c26f4d75987772457a25d61.jpg)  
(a) Scaling of  $P_{1 / 2}$  with  $\alpha$

![](images/8d62d7c3094e49f99d222ad6a550906c85a46103736b7d1a1d7e2d9b1c71e7d4.jpg)  
Figure 4: Critical sample size  $P_{1/2}$  measures the onset of the variance limited regime as a function of  $\alpha$  at fixed  $N$ . (a) More feature learning (small  $\alpha$ ) delays transition to the variance limited regime. (b)  $P_{1/2}$  as a function of  $N$  for fixed  $\alpha$  has roughly  $P_{1/2} \sim \sqrt{N}$  scaling.  
(b) Scaling of  $P_{1 / 2}$  with  $N$

the  $\mathrm{eNTK}_0$  predictor, or a sufficiently lazy network trained to interpolation on a dataset  $\mathcal{D}$ . Then,

$$
\left\langle \left(f _ {\theta_ {0}, \mathcal {D}} ^ {*} (\boldsymbol {x}) - y\right) ^ {2} \right\rangle_ {\theta_ {0}, \mathcal {D}, \boldsymbol {x}, y} = \left\langle \left(f _ {\mathcal {D}} ^ {\infty} (\boldsymbol {x}) - y\right) ^ {2} \right\rangle_ {\mathcal {D}, \boldsymbol {x}, y} + O (1 / N). \tag {2}
$$

We prove this equality using a relationship between the infinite-width network and an infinite ensemble of finite-width networks derived in Appendix B. There we also show that the  $O(1 / N)$  term is strictly positive for sufficiently large  $N$ . Thus, for lazy networks, finite width effects lead to strictly worse generalization error. The decomposition in Equation 2 continues to hold for rich networks at small  $\alpha$  if  $f^{\infty}$  is interpreted as the infinite-width mean field limit. In this case one can show that ensembles of rich networks are approximating an infinite width limit in the mean-field regime. See Appendix B for details.

# 3.3 FEATURE LEARNING DELAYS VARIANCE LIMITED TRANSITION

We now consider how feature learning alters the onset of the variance limited regime, and how this onset scales with  $\alpha, N$ . We define the onset of the variance limited regime to take place at the value  $P^{*} = P_{1/2}$  where over half of the generalization error is due to variance over initializations. Equivalently we have  $E_{g}(\tilde{f}^{*}) / E_{g}(f^{*}) = 1/2$ . By using an interpolation method together with bisection, we solve for  $P_{1/2}$  and plot it in Figure 4.

Figure 4b shows that  $P_{1/2}$  scales as  $\sqrt{N}$  for this task. In the next section, we shall show that this scaling is governed by the fact that  $P_{1/2}$  is close to the value where the infinite width network generalization curve  $E_g^\infty$  is equal to the variance of the final kernel. In this case the quantities to compare are  $E_g^\infty \approx P^{-2}$  and  $\mathrm{Var~eNTK}_f \approx N^{-1}$ .

We can understand the delay of the variance limited transition, as well as the lower value of the final plateau using a mechanistic picture similar to the effect observed in Atanasov et al. (2021). In that setting, under small initialization, the kernel follows a deterministic trajectory, picking up a low rank component in the direction of the train set targets  $\boldsymbol{yy}^{\top}$ , and then changing only in scale as the network weights grow to interpolate the dataset. In their case, for initial output scale  $\sigma^L$ , the final kernel is deterministic up to a variance of  $O(\sigma)$ . In our case, the kernel variance at initialization scales as  $\sigma^{2L} / N$ . As  $\sigma \to 0$  the kernel's trajectory becomes deterministic up to a variance term scaling with  $\sigma$  as  $O(\sigma)$ , which implies that the final predictor also has a variance scaling as  $O(\sigma)$ .

# 4 SIGNAL PLUS NOISE CORRELATED FEATURE MODEL

In Section 3.2 we have shown that in both the rich and lazy regimes, the generalization error of the NN is well approximated by the generalization of a kernel regression solution with  $\mathrm{eNTK}_f$ . This

![](images/9f3112d41b965324e6f1f51fc353f0419994098ea56248a5fa6476d44e36fe20.jpg)  
(a) Ensembling Methods

![](images/01a2fff8f129c31d5bda61d41c20d747bb9322d943a06e9757a3d650ec187964.jpg)  
Figure 5: The random feature model suggests three possible types of ensembling: averaging the output function  $f(\pmb{x},\theta)$ , averaging the final NTK  $K(\pmb{x},\pmb{x}';\theta)$ , and averaging the induced features  $\psi(\pmb{x},\theta)$ . We analyze these ensembling methods for a  $k = 1$  task with a width  $N = 100$  ReLU network. (a) While all three ensembling methods improve generalization, averaging either the kernel  $\langle K\rangle$  or features  $\langle \psi \rangle$  gives a better improvement to generalization than averaging the output function  $\langle f\rangle$ . This indicates that computing final kernels for many richly trained networks and performing regression with this averaged kernel gives the best performance. (b) We plot the relative error of each ensembling method against the single init neural network. The gap between ensembling and the single init NN becomes evident for sufficiently large  $P \sim P_{1/2}$ . For small  $\alpha$ , all ensembling methods perform comparably, while for large  $\alpha$  ensembling the kernel or features gives much lower  $E_g$  than averaging the predictors.  
(b) Reduction in  $E_{g}$  for Each Ensembling Technique

finding motivates an analysis of the generalization of kernel machines which depend on network initialization  $\theta_0$ . Unlike many analyses of random feature models which require high dimensional Gaussian random data, and focus on two layer networks (Mei & Montanari, 2022; Adlam & Pennington, 2020a; Gerace et al., 2020; Ba et al., 2022), we propose to analyze regression with the eNTK  $f$  for an arbitrary depth network on an arbitrary data distribution. This work builds on the kernel generalization theory for non-random kernels developed with statistical mechanics (Bordelon et al., 2020; Canatar et al., 2021; Simon et al., 2021). We will attempt to derive approximate learning curves in terms of the final NTK's signal and noise components, which provide some phenomenological explanations of the onset of the variance limited regime and the benefits of feature learning. Starting with the final NTK  $K_{\theta_0}(\pmb{x},\pmb{x}')$  which depends on the random initial parameters  $\theta_0$ , we project its square root  $K_{\theta_0}^{1/2}(\pmb{x},\pmb{x}')$  on a fixed orthonormal (with respect to  $p(\pmb{x})$ ) basis  $\{b_k(x)\}_{k=1}^{\infty}$  to define a feature map

$$
\psi_ {k} (\boldsymbol {x}, \theta) = \int d \boldsymbol {x} ^ {\prime} p \left(\boldsymbol {x} ^ {\prime}\right) K _ {\theta_ {0}} ^ {1 / 2} \left(\boldsymbol {x}, \boldsymbol {x} ^ {\prime}\right) b _ {k} \left(\boldsymbol {x} ^ {\prime}\right), k \in \{1, \dots , \infty \}. \tag {3}
$$

The kernel can be reconstructed from these features  $K_{\theta_0}(\pmb{x}, \pmb{x}') = \sum_k \psi_k(\pmb{x}, \theta_0) \psi_k(\pmb{x}', \theta_0)$ . The kernel interpolation problem can be solved by performing linear regression with features  $\psi(\pmb{x}, \theta_0)$ . Here,  $\pmb{w}(\theta_0) = \lim_{\lambda \to 0} \operatorname{argmin}_{\pmb{w}} \sum_{\mu=1}^{P} [\pmb{w} \cdot \psi(\pmb{x}_\mu, \theta_0) - y_\mu]^2 + \lambda |\pmb{w}|^2$ . The learned function  $f(\pmb{x}, \theta_0) = \pmb{w}(\theta_0) \cdot \psi(\pmb{x}, \theta_0)$  is the minimum norm interpolator for the kernel  $K(\pmb{x}, \pmb{x}'; \theta_0)$ , and from 3 would match the neural network learning curve. Since the target function  $y$  does not depend on the initialization  $\theta_0$ , we decompose it in terms of the mean features  $\bar{\psi}(\pmb{x}) = \langle \psi(\pmb{x}, \theta_0) \rangle_{\theta_0}$  as  $y(\pmb{x}) = \pmb{w}^* \cdot \bar{\psi}(\pmb{x})$ . For convenience, we define the fluctuation of the feature as  $\delta(\pmb{x}, \theta_0) = \psi(\pmb{x}, \theta_0) - \bar{\psi}(\pmb{x})$ . The generalization error depends on signal  $\pmb{\Sigma}_s$  and noise  $\pmb{\Sigma}_n$  correlation matrices

$$
\boldsymbol {\Sigma} _ {s} = \left\langle \bar {\psi} (\boldsymbol {x}) \bar {\psi} (\boldsymbol {x}) ^ {\top} \right\rangle_ {\boldsymbol {x}}, \boldsymbol {\Sigma} _ {n} = \left\langle \boldsymbol {\delta} (\boldsymbol {x}, \theta_ {0}) \boldsymbol {\delta} (\boldsymbol {x}, \theta_ {0}) ^ {\top} \right\rangle_ {\boldsymbol {x}, \theta_ {0}}. \tag {4}
$$

Using the completeness of  $\{b_k\}$ , we decouple the initialization and sample dependence in the fluctuation  $\delta(\boldsymbol{x}, \theta_0) = \Sigma_n^{1/2} \boldsymbol{A}(\theta_0) \boldsymbol{b}(\boldsymbol{x})$ , where  $\boldsymbol{A}$  is a mean zero with  $\left\langle \boldsymbol{A}(\theta_0) \boldsymbol{A}(\theta_0)^\top \right\rangle_{\theta_0} = \boldsymbol{I}$ .

In this random feature model, one can interpret the fluctuations in  $K(\pmb{x}, \pmb{x}'; \theta_0)$  as generating fluctuations in the features  $\psi(\pmb{x}, \theta_0)$  which induce fluctuations in the learned network predictor  $f(\pmb{x}, \theta_0)$ . To illustrate the relative improvements to generalization from denoising these three different objects, in Figure 5, we compare

- averaging the final kernel  $K$  and performing kernel regression with  $\langle K \rangle$ ,

- averaging the induced features  $\psi$  and performing linear regression with  $\langle \psi \rangle$ ,  
- averaging network predictions  $f$  directly.

For all  $\alpha$ , all ensembling methods provide improvements over training a single NN. However, we find that averaging the kernel directly and performing regression with this kernel exhibits the largest reduction in generalization error. Averaging features performs comparably. However, ensemble averaging network predictors does not perform as well as either of these other two methods. The gap between these ensembling methods is more significant in the lazy regime (large  $\alpha$ ) and is negligible in the rich regime (small  $\alpha$ ).

# 4.1 TOY MODEL AND APPROXIMATE LEARNING CURVES

Working under the approximation that  $A(\theta_0)$ 's entries are iid Gaussian over the disorder in  $\theta_0$ , and assuming that  $P$  is much smaller than the total parameter count of the network, we obtain a modification the noise-free learning curves of Bordelon et al. (2020) and Canatar et al. (2021) which depend on  $\{\pmb{w}^*, \pmb{\Sigma}_s, \pmb{\Sigma}_n, P\}$ . In this model, the generalization error will plateau to  $E_g \sim \pmb{w}^*\pmb{\Sigma}_n[\pmb{\Sigma}_n + \pmb{\Sigma}_s]^{-1}\pmb{\Sigma}_s\pmb{w}^*$ . Under the further assumption of codiagonalizable signal  $\pmb{\Sigma}_s = \sum_k \lambda_k \pmb{u}_k \pmb{u}_k^\top$  and noise  $\pmb{\Sigma}_n = \sum_k \eta_k \pmb{u}_k \pmb{u}_k^\top$  correlations, the estimation error along the  $k$ -th eigenfunction is

$$
E _ {k} (P) \sim \frac {1}{1 + P \left(\lambda_ {k} + \eta_ {k}\right) \kappa^ {- 1}} \left[ \frac {1 + \eta_ {k} P \kappa^ {- 1}}{1 + P \kappa^ {- 1} \left(\lambda_ {k} + \eta_ {k}\right)} + P \kappa^ {- 1} \eta_ {k} \right], \tag {5}
$$

where  $\kappa$  solves the implicit equation  $1 = \sum_{k} \frac{\lambda_k + \eta_k}{(\lambda_k + \eta_k)P + \kappa}$ . This  $k$ -th mode error plateaus to  $\frac{\eta_k}{\lambda_k + \eta_k}$  which can be interpreted in terms of a signal to noise ratio  $\frac{\lambda_k}{\eta_k}$ . While the first term decays like  $P^{-1}$ , the second term approaches a constant at large  $P$ , giving the asymptotic error. Details of these calculations are given in Appendix D.

# 4.2 EXPLAINING FEATURE LEARNING BENEFITS AND ERROR PLATEAUS

Using this theory, we can attempt to explain some of the observed phenomena associated with the onset of the variance limited regime. First, we note that the noise correlations  $\Sigma_{n}$  should scale as  $O(1 / N)$  since kernels, either in the lazy or rich regime, exhibit variance  $1 / N$  fluctuations over initialization. In Figure 6 (a), we show learning curves for networks of different widths in the lazy regime. Small width networks enter the variance limited regime earlier and have higher error. Similarly, if we alter the scale of the noise in our toy model, the corresponding transition time  $P_{1 / 2}$  is smaller and the asymptotic error is higher. In Figure 6 (c), we show that our theory also predicts the onset of the variance limited regime at  $P_{1 / 2} \sim \sqrt{N}$ . We again stress that this scaling is a consequence of the structure of the task. Since the target function is an eigenfunction of the kernel, the infinite width error goes as  $1 / P^2$ . Since the variance scales as  $1 / N$ , these two quantities become comparable at  $P \sim \sqrt{N}$ . Often, more complicated tasks exhibit power law decays where  $E_g^{N = \infty} = P^{-\beta}$  with  $\beta < 2$  (Spigler et al., 2020; Bahri et al., 2021). For such tasks, we expect a transition around  $P_{1 / 2} \sim N^{1 / \beta}$ . For tasks that are harder to learn, namely those that have infinite-width generalization error scaling slower than  $P^{-1 / 2}$ , the variance onset can occur after the transition to the underparameterized regime. We leave it to future study to explore datasets with such scaling laws.

Using our model, we can also approximate the role of feature learning as enhancement in the signal correlation along task-relevant eigenfunctions. In Figure 6 (a) we plot the learning curves for networks trained with different levels of feature learning, controlled by  $\alpha$ . We see that feature learning (small  $\alpha$ ) leads to improvements in the learning curve both before and after the variance limited regime begins. In Figure 6 (b), we plot the theoretical generalization for kernels with enhanced signal eigenvalue for the task eigenfunction  $y(\pmb{x}) = \phi_k(\pmb{x})$ . This enhancement, based on the intuition of kernel alignment, leads to lower bias and lower asymptotic variance. However, this model does not capture the fact that feature learning advantages are small at small  $P$  and that the slopes of the learning curves are different at different  $\alpha$ . Following the observation of Paccolat et al. (2021a) that kernel alignment can occur with scale  $\sqrt{P}$ , we plot the learning curves for signal enhancements that scale as  $\sqrt{P}$ . This yields better qualitative agreement to the network's learning curve.

![](images/84d67683cbc010e5a0d11ada6e3835a198573903e5f6a1c6ffd6930aa36b39d3.jpg)  
(a)  $E_g^{NN}$  for different  $N$

![](images/1654ccc72e25f1e80896b3141171a3072e744e129ac48ca957cf6b8b90832200.jpg)  
(b) Small  $N\approx$  Large  $\sigma^2$

![](images/20b6d45b8d0cf94e8ad9e7dffd38d9a63696cfa08a8f422dbdec28d43dbb22db.jpg)  
(c) Variance Limited Transition

![](images/212658deddac1216aeea1cc5b3d15c514f66eeeed44fa7ce1b99f45291cb2d09.jpg)  
(d) Feature Scalings

![](images/dfe576bef10b7f9d73f18a1e816d31506220e6c3cd055f04b07d92753cbcf96f.jpg)  
(e) Richness  $\approx$  Amplified  $\pmb{\Sigma}_{s}$

![](images/45563c3ce88d522b7e10a909af3ee693f0b2c81d5702ec6abc3bdf20bd17b9d8.jpg)  
Figure 6: Our toy model of noisy features reproduces qualitative dependence of learning curves on kernel fluctuations and feature learning. (a) The empirical learning curves for networks of varying width  $N$  at large  $\alpha$ . (b) Noisy kernel regression learning curve with noise eigenvalues  $\eta_{k} = \sigma^{2}\lambda_{k}$ . The eigenvalues  $\lambda_{k}$  were taken from the infinite width NTK for our depth 3 ReLU network. The model predicts an asymptote at  $E_{k} \sim \frac{1}{\sigma^{2}}$ . (c) This toy model reproduces the approximate scaling of the transition sample size  $P_{1/2} \sim N^{1/2}$ . (d) Width  $N = 1000$  networks trained with varying richness  $\alpha$ . Small  $\alpha$  improves the early learning curve and late stage learning curve. (e) Theory curves for a kernel with amplified eigenvalue  $\lambda_{k} \rightarrow \lambda_{k} + \Delta \lambda_{k}$  corresponding to the target eigenfunction. This amplification mimics the effect of enhanced kernel alignment in the low  $\alpha$  regime. Increasing this amplification factor improves the early generalization performance and the asymptote. (f) A model of  $P$ -dependent alignment where  $\Delta \lambda_{k} \sim \sqrt{P}$  gives a better qualitative match to (d).  
(f)  $P$  -dependent Amplification

Though this toy model reproduces the onset of the variance limited regime  $P_{1/2}$  and the reduction in variance due to feature learning, our current result is not the complete story. First, it is unclear if modeling  $A(\theta)$  as a Gaussian matrix is appropriate. For instance  $A$  could be zero off-diagonal and just have diagonal entries. Second, we assume that parameter count is much larger than  $P$  so our current equations would not capture the interesting effects at the interpolation threshold (Adlam & Pennington, 2020b). Lastly, in the lazy regime, the neural network learning curves do not asymptote as quickly as our model predicts. We leave these problems for future investigation.

# 5 CONCLUSION

We performed an extensive empirical study for deep ReLU NNs learning a fairly simple polynomial regression problems. For sufficiently large dataset size  $P$ , all neural networks under-perform the infinite width limit, and we demonstrated that this worse performance is driven almost entirely by initialization variance. We show that the onset of the variance limited regime can occur relatively early with scaling  $P_{1/2} \sim \sqrt{N}$ , but this can be delayed by enhancing feature learning. Finally, we have proposed a simple random-feature model to attempt to explain these effects and qualitatively reproduce the observed behavior, as well as quantitatively reproducing the relevant scaling relationship for  $P_{1/2}$ . This work takes a step towards understanding scaling laws in regimes where finite-size networks undergo feature learning. This has implications for how the choice of initialization scale, neural architecture, and number of ensembles of a network can be tuned to achieve optimal performance under a fixed compute and data budget.

# REFERENCES

Ben Adlam and Jeffrey Pennington. The neural tangent kernel in high dimensions: Triple descent and a multi-scale theory of generalization. In International Conference on Machine Learning, pp. 74-84. PMLR, 2020a.  
Ben Adlam and Jeffrey Pennington. Understanding double descent requires a fine-grained biasvariance decomposition. Advances in neural information processing systems, 33:11022-11032, 2020b.  
Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners: The silent alignment effect. In International Conference on Learning Representations, 2021.  
Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. High-dimensional asymptotics of feature learning: How one gradient step improves the representation. arXiv preprint arXiv:2205.01445, 2022.  
Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws, 2021. URL https://arxiv.org/abs/2102.06701.  
Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. arXiv preprint arXiv:2205.09653, 2022.  
Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1024-1034. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/bordelon20a.html.  
James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, et al. Jax: composable transformations of python+ numpy programs. Version 0.2, 5:14-24, 2018.  
Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks. Nature Communications, 12, 2021.  
Lenaïc Chizat, Edouard Oyallon, and Francis R. Bach. On lazy training in differentiable programming. In NeurIPS, 2019.  
Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. The Journal of Machine Learning Research, 13(1):795-828, 2012.  
Stéphane d'Ascoli, Levent Sagun, and Giulio Biroli. Triple descent and the two kinds of overfitting: Where & why do they appear? Advances in Neural Information Processing Systems, 33:3058-3069, 2020.  
Oussama Dhifallah and Yue M Lu. A precise performance analysis of learning with random features. arXiv preprint arXiv:2008.11904, 2020.  
Ethan Dyer and Guy Gur-Ari. Asymptotics of wide networks from feynman diagrams. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1gFvANKDS.  
Stéphane d'Ascoli, Maria Refinetti, Giulio Biroli, and Florent Krzakala. Double trouble in double descent: Bias and variance (s) in the lazy regime. In International Conference on Machine Learning, pp. 2280-2290. PMLR, 2020.  
Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M Roy, and Surya Ganguli. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 5850-5861. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/405075699f065e43581f27d67bb68478-Paper.pdf.

Mario Geiger, Arthur Jacot, Stefano Spigler, Franck Gabriel, Levent Sagun, Stéphane d'Ascoli, Giulio Biroli, Clément Hongler, and Matthieu Wyart. Scaling description of generalization with number of parameters in deep learning. Journal of Statistical Mechanics: Theory and Experiment, 2020(2):023401, 2020a.  
Mario Geiger, Stefano Spigler, Arthur Jacot, and Matthieu Wyart. Disentangling feature and lazy training in deep neural networks. Journal of Statistical Mechanics: Theory and Experiment, 2020 (11):113301, 2020b.  
Federica Gerace, Bruno Loureiro, Florent Krzakala, Marc Mészard, and Lenka Zdeborová. Generalisation error in learning with random features and the hidden manifold model. In International Conference on Machine Learning, pp. 3452-3462. PMLR, 2020.  
Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. When do neural networks outperform kernel methods? In NeurIPS, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/a9df2255ad642b923d95503b9a7958d8-Abstract.html.  
Boris Hanin and Mihai Nica. Finite depth and width corrections to the neural tangent kernel. In International Conference on Learning Representations, 2019.  
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.  
Hong Hu and Yue M Lu. Universality laws for high-dimensional learning with random features. arXiv preprint arXiv:2009.07669, 2020.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: convergence and generalization in neural networks (invited paper). Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, 2018.  
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, pp. 3519-3529. PMLR, 2019.  
Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jascha Sohl-Dickstein. Wide neural networks of any depth evolve as linear models under gradient descent. ArXiv, abs/1902.06720, 2019.  
Philip M. Long. Properties of the after kernel. CoRR, abs/2105.10585, 2021. URL https://arxiv.org/abs/2105.10585.  
Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75(4):667-766, 2022.  
Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences, 115(33):E7665-E7671, 2018.  
Roman Novak, Lechao Xiao, Jiri Hron, Jaehoon Lee, Alexander A. Alemi, Jascha Sohl-Dickstein, and Samuel S. Schoenholz. Neural tangents: Fast and easy infinite neural networks in python. In International Conference on Learning Representations, 2020. URL https://github.com/google/neural-tangents.  
Guillermo Ortiz-Jiménez, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. What can linearized neural networks actually say about generalization? Advances in Neural Information Processing Systems, 34:8998-9010, 2021.  
Jonas Paccolat, Leonardo Petrini, Mario Geiger, Kevin Tyloo, and Matthieu Wyart. Geometric compression of invariant manifolds in neural networks. Journal of Statistical Mechanics: Theory and Experiment, 2021(4):044001, apr 2021a. doi: 10.1088/1742-5468/abf1f3. URL https://doi.org/10.1088/1742-5468/abf1f3.

Jonas Paccolat, Leonardo Petrini, Mario Geiger, Kevin Tyloo, and Matthieu Wyart. Geometric compression of invariant manifolds in neural networks. Journal of Statistical Mechanics: Theory and Experiment, 2021(4):044001, 2021b.  
Daniel A. Roberts, Sho Yaida, and Boris Hanin. The principles of deep learning theory, 2021.  
James B. Simon, Madeline Dickens, and Michael R. DeWeese. Neural tangent kernel eigenvalues accurately predict generalization, 2021.  
Stefano Spigler, Mario Geiger, and Matthieu Wyart. Asymptotic learning curves of kernel methods: empirical data versus teacher-student paradigm. Journal of Statistical Mechanics: Theory and Experiment, 2020(12):124001, 2020.  
Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105-6114. PMLR, 2019.  
Nikhil Vyas, Yamini Bansal, and Preetum Nakkiran. Limitations of the ntk for understanding generalization in deep learning. arXiv preprint arXiv:2206.10012, 2022.  
Alexander Wei, Wei Hu, and Jacob Steinhardt. More than a toy: Random matrix models predict how real-world neural representations generalize. arXiv preprint arXiv:2203.06176, 2022.  
Greg Yang and Edward J. Hu. Feature learning in infinite-width neural networks. ArXiv, abs/2011.14522, 2020.
