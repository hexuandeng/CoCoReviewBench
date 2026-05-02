# THE LARGE LEARNING RATE PHASE OF DEEP LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The choice of initial learning rate can have a profound effect on the performance of deep networks. We present empirical evidence that networks exhibit sharply distinct behaviors at small and large learning rates. In the small learning rate phase, training can be understood using the existing theory of infinitely wide neural networks. At large learning rates, we find that networks exhibit qualitatively distinct phenomena that cannot be explained by existing theory: The loss grows during the early part of training, and optimization eventually converges to a flatter minimum. Furthermore, we find that the optimal performance is often found in the large learning rate phase. To better understand this behavior we analyze the dynamics of a two-layer linear network and prove that it exhibits these different phases. We find good agreement between our analysis and the training dynamics observed in realistic deep learning settings.

# 1 INTRODUCTION

Deep learning has shown remarkable success across a variety of tasks. At the same time, our theoretical understanding of deep learning methods remains limited. In particular, the interplay between training dynamics, properties of the learned network, and generalization remains a largely open problem.

In tackling this problem, much progress has been made by studying deep neural networks whose hidden layers are wide. In the limit of infinite width, connections between stochastic gradient descent (SGD) dynamics of neural networks, compositional kernels, and linear models have been made. These connections hold when the learning rate is sufficiently small. However, a theory of the dynamics of deep networks that operate outside this regime remains largely open.

In this work, we present evidence that SGD dynamics change significantly when the learning rate is above a critical value,  $\eta_{\mathrm{crit}}$ , determined by the local curvature of the loss landscape at initialization. These dynamics are stable above the critical learning rate, up to a maximum learning rate  $\eta_{\mathrm{max}}$ . Training at these large learning rates results in different signatures than observed for learning rates  $\eta < \eta_{\mathrm{crit}}$ : the loss initially increases and peaks before decreasing again, and the local curvature drops significantly early in training. We typically find that the best performance is obtained when training above the critical learning rate. Empirically, we find these two learning rate regimes are robust, holding across a variety of architectural and data settings.

Figure 1 highlights our key findings. We now describe the main contributions of this work.

# 1.1 TRAINING WITH A LARGE LEARNING RATE LEADS TO A CATAPULT EFFECT

Consider a deep network defined by the network function  $f(\theta, x)$ , where  $\theta$  are the model parameters and  $x$  the input. We define the curvature  $\lambda_t$  at training step  $t$  to be the max eigenvalue of the Fisher Information Matrix,  $F_t \coloneqq \mathbb{E}_x\left[\nabla_\theta f(\theta_t,x)\nabla_\theta f(\theta_t,x)^T\right]$  Amari et al. (2000); Karakida et al. (2018). Equivalently,  $\lambda_t$  is the max eigenvalue of the Neural Tangent Kernel Jacot et al. (2018).

Figure 2 shows the results of training several deep networks with mean squared error (MSE) loss using SGD with a range of learning rates. The loss and curvature are measured at every step during

![](images/8c83dc2011100f5415e518ffc7b32d0859a2b373bf4263e137eb41b35356c375.jpg)  
Figure 1: Large learning rates lead to large weight movement and better performance. (a) A visualization of gradient descent dynamics derived in our analytic model. A 2D slice of parameter space is shown, where lighter color indicates higher loss and dots represent points visited during optimization. Initially, the loss grows rapidly while local curvature decreases. Once curvature is sufficiently low, gradient descent converges to a flat minimum. We call this the catapult effect. See Figures S2 and S1 for more details. (b) Confirmation of our predictions in a practical deep learning setting. Line shows the test accuracy of a Wide ResNet trained on CIFAR-10 as a function of learning rate, each trained for a fixed number of steps. Dashed lines show our predictions for the boundaries of the large learning rate regime (the catapult phase), where we expect optimal performance to occur. Maximal performance is achieved between the dashed lines, confirming our predictions. See Section 2 for details.

![](images/39b6599ac4eff4aa4008f249f17a3912ff6ac6b1451ec1681985d5364122f6f6.jpg)

the early part of training. We notice the following effects, which occur when the learning rate is above the critical value  $\eta_{\mathrm{crit}} = 2 / \lambda_0$ , where  $\lambda_0$  is the curvature at initialization.

1. During the first few steps of training, the loss grows significantly compared to its initial value before it begins decreasing. We call this the catapult effect.  
2. Over the same time frame, the curvature decreases until it is below  $2 / \eta$

We can build intuition for these effects using loss landscape considerations. Consider a linear model where the curvature of the loss landscape is given by  $\lambda_0$ . Here, curvature means the largest eigenvalue of the linear model kernel. The model can be trained using gradient descent as long as the learning rate  $\eta$  obeys  $\eta < 2 / \lambda_0$ . When  $\eta > 2 / \lambda_0$ , the loss diverges and optimization fails.

Next, consider a deep network. If we train the model with learning rate  $\eta >\eta_{\mathrm{crit}}$ , we may again expect the loss to grow initially, assuming the curvature is approximately constant in the neighborhood of the initial point in parameter space. This is the effect observed in Figure 2. However, unlike the linear case, optimization may still succeed if gradient descent is able to navigate to an area of the landscape that has lower curvature  $\lambda$ , such that  $\eta < 2 / \lambda$ . This is indeed what we observe in practice.

In Figure 1 we show that optimal performance typically occurs when a network is trained in the large learning rate regime. As discussed further in Section 2, this is true even when the compute budget for smaller learning rates is increased to account for the smaller step size. This is consistent with previous observations in the literature, which showed a correlation between performance and the flatness of the minimum (Keskar et al., 2016).

# 1.2 AT LARGE WIDTH, A SHARP DISTINCTION BETWEEN LEARNING RATES REGIMES

The large width limit of deep networks has been shown to lead to simplified training dynamics that are amenable to theoretical study, as in the case of the Neural Tangent Kernel (Jacot et al., 2018). In this work we show that the distinction between small and large learning rates becomes sharply defined at large width. This can be seen in Figures 2c, 2f, which show the curvature of sufficiently wide networks after the initial part of training, as a function of learning rate. When  $\eta < \eta_{\mathrm{crit}}$  the curvature is approximately independent of the learning rate, while for  $\eta > \eta_{\mathrm{crit}}$  the curvature is lower than  $2 / \eta$ .

![](images/9b28c3f9cf56eb2ad3c136e352511ebb16b415d29b06ea14cb0ac36b81010bd7.jpg)  
(a)

![](images/b63710e2a7eb057a056b09cfc39981adf4cc9dcb501d0d721559165cb21aeb2b.jpg)  
(b)

![](images/2a30c0d17431df753c8df35611430f901f625ea4e9516da42a9e8a36e482945f.jpg)  
(c)

![](images/f42cea3606d4fd3acfd3baa46220bb4a7d2a49be097fb02bf7307890b4beef7f.jpg)  
(d)

![](images/2d84230b33023a8abd11b45ec673bf502067f08cb6d1d50ed8092413ba89fd01.jpg)  
Figure 2: Early time catapult dynamics. (a,b,c) A 3 hidden layer fully-connected network with ReLU nonlinearity with width 2048 trained on MNIST ( $\eta_{\mathrm{crit}} = 6.25$ ). (d,e,f) Wide ResNet 28-10 trained on CIFAR-10 ( $\eta_{\mathrm{crit}} = 0.18$ ). Both networks are trained with vanilla SGD; for more experimental details see Appendix A. (a,d) Early time dynamics of the training loss for learning rates in the linear and catapult phases. (b) Maximum value of the loss as a function of the learning rate. (e) Early time dynamics of the curvature for learning rates in the linear and catapult phase. (c,f)  $\lambda_t$  measured at  $t \cdot \eta = 250$  (for FC) and  $t \cdot \eta = 30$  (for WRN), as a function of learning rate. Training diverges for learning rates in the shaded region.  
(e)

![](images/8741789e787d8f35dacd68ddb7a8062ec1b2b9f28452543133d6fc1ed544bb07.jpg)  
(f)

In Section 3 we analyze the gradient descent dynamics of 2-layer linear networks, and find that they exhibit similar behavior that can be understood in the large width limit. When training with small learning rates, existing theory can describe the training dynamics of such networks. And, we present new theoretical results that explain the behavior at large learning rates. To summarize our findings, in the large width limit we identify two separate learning rate regimes, or phases, with the following characteristics.

Lazy phase:  $\eta < 2 / \lambda_0$ . For sufficiently small learning rate, the curvature  $\lambda_{t}$  at training step  $t$  remains constant throughout training, and the model becomes equivalent to a linear model (Jacot et al., 2018; Lee et al., 2019). The model converges to a nearby point in parameter space, and this behavior is sometimes called lazy training (Du et al., 2019; Zou et al., 2018; Allen-Zhu et al., 2019; Li & Liang, 2018; Chizat et al., 2019).

**Catapult phase:**  $\eta_{\mathrm{crit}} < \eta < \eta_{\mathrm{max}}$ . At large learning rates the loss grows to be of order the width  $n$  over a number of training steps that is of order  $\log (n)$ . During the same period, the curvature decreases until it is below  $2 / \eta$ . Beyond this point the loss decreases and training converges, ultimately reaching a flat minimum (relative to that reached in the lazy phase). The gradient descent dynamics in this phase are visualized in Figure 1 and in Figure S1.

The maximum learning rate  $\eta_{\mathrm{max}}$  (beyond which training no longer converges) depends on the setup. In our theoretical model the maximum learning rate is  $\eta_{\mathrm{max}} = 4 / \lambda_0$ . For ReLU networks, we find empirically that  $\eta_{\mathrm{max}} \approx 12 / \lambda_0$ .

# 1.3 LIMITATIONS

Our empirical results and our theoretical analysis focus on the case of MSE loss and on training with vanilla SGD, and do not extend to the case of cross-entropy loss, or to training with other optimizers such as momentum. Additionally, our theoretical analysis involves a 2-layer linear network, and does not apply to the case of networks with non-linearities such as ReLU.

# 1.4 RELATED WORKS

Our work builds on several existing results, which we now briefly review.

The existing theory of infinite width networks is insufficient to describe large learning rates. A recent body of work has investigated the gradient descent dynamics of deep networks in the limit of infinite width (Daniely, 2017; Jacot et al., 2018; Lee et al., 2019; Du et al., 2019; Zou et al., 2018; Allen-Zhu et al., 2019; Li & Liang, 2018; Chizat et al., 2019; Mei et al., 2018; Rotskoff & Vanden-Eijnden, 2018; Sirignano & Spiliopoulos, 2018; Woodworth et al., 2019; Naveh et al.; Xiao et al., 2019). Of particular relevance is the work by Jacot et al. (2018) showing that gradient flow in the space of functions is governed by a dynamical quantity called the Neural Tangent Kernel (NTK) which is fixed at its initial value in this limit. Lee et al. (2019) showed this result is equivalent to training the linearization of a model around its initialization in parameter space. Finally, moving away from the strict limit of infinite width by working perturbatively, Dyer & Gur-Ari (2020); Huang & Yau (2019) introduced an approach to computing the finite-width corrections to network evolution.

Despite this progress, in many practical deep learning settings, the neural network has finite width and evolves nontrivially, with a large change in its associated Neural Tangent Kernel. Depending on the architecture and hyperparameters, such networks may give superior performance. Prior work has compared the performance of finite-width, SGD-trained deep networks with the infinite-width kernels derived from the networks (Lee et al., 2018; Novak et al., 2019; Arora et al., 2019). Performance gaps are observed in some cases, notably in convolutional networks, implying that existing infinite-width theory is insufficient to explain the performance of deep networks in such settings where the network evolves nontrivially.

Large learning rate SGD improves generalization. SGD training with large initial learning rates often leads to improved performance over training with small initial learning rates (see (Li et al., 2019; Leclerc & Madry, 2020; Xie et al., 2020; Frankle et al., 2020; Jastrzebski et al., 2020) for recent discussions). It has been suggested that one of the mechanisms underlying the benefit of large learning rates is that noise from SGD leads to flat minima, and that flat minima generalize better than sharp minima (Hochreiter & Schmidhuber, 1997; Keskar et al., 2016; Smith & Le, 2018; Jiang et al., 2020; Park et al., 2019) (though see Dinh et al. (2017) for discussion of some caveats). According to this suggestion, training with a large learning rate (or with a small batch size) can improve performance because it leads to more stochasticity during training (Smith & Le, 2018; Mandt et al., 2017; Smith et al., 2017; Smith et al., 2018).

We develop a connection between large learning rate and flatness of minima in models trained via SGD. Unlike the relationship explored in most previous work though, this connection is not driven by SGD noise, but arises solely as a result of training with a large initial learning rate, and holds even for full batch gradient descent.

# 2 EXPERIMENTAL RESULTS

In a variety of deep learning settings, we find clear evidence of the different phases introduced in Section 1. The experiments all use MSE loss, sufficiently wide networks, and vanilla SGD with learning rate  $\eta$ . Parameters such as network architecture, choice of non-linearity, weight parameterization, and regularization, do not significantly affect this conclusion.

In these experiments, we define the curvature  $\lambda$  as the maximum eigenvalue of the Fisher Information Matrix or, equivalently, as the maximum eigenvalue of the Neural Tangent Kernel (NTK). Given a network function  $f: \mathbb{R}^d \to \mathbb{R}$  with model parameters  $\theta \in \mathbb{R}^p$ , and a training set  $\{(x_{\alpha}, y_{\alpha})\}_{\alpha=1}^{m}$ , the NTK  $\Theta: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$  is defined by  $\Theta(x, x') := \frac{1}{m} \sum_{\mu=1}^{p} \nabla_{\theta} f(x)^T \nabla_{\theta} f(x')$ . We denote the curvature at time  $t$  by  $\lambda_t$ , equal to the maximum eigenvalue of  $\Theta$ . Another common measure of local curvature is the maximum Hessian eigenvalue; at large width we expect these measures to agree (Dyer & Gur-Ari, 2020), and we verify the agreement in Appendix D.6.

Building on the observed correlation between flat minima and generalization performance (Keskar et al., 2016; Jiang et al., 2020), we conjecture that optimal performance occurs in the large learning rate (catapult) phase, where optimization converges to a low curvature minimum. For a fixed amount of computational budget, we find that this conjecture holds in all cases we tried. Even when comparing different learning rates trained for a fixed amount of physical time  $t_{\mathrm{phys}} = t \cdot \eta$ , we find that performance of models trained in the catapult phase either matches or exceeds that of models trained with learning rates below  $\eta_{\mathrm{crit}}$ .

# 2.1 EARLY TIME CURVATURE DYNAMICS

Here we present empirical support for the lazy and catapult phases of training described in the previous section. Additional experimental results are presented in the appendix. We find that the lazy phase is characterized by small changes to the curvature and loss during training, while the catapult exhibits large deviations.

Figure 2 shows the curvature during the early part of training for two deep learning settings that involve sufficiently wide networks. The results are compared against the prediction of a phase transition at  $\eta_{\mathrm{crit}} = 2 / \lambda_0$ . For learning rates  $\eta < \eta_{\mathrm{crit}}$  (lazy phase), the curvature is independent of the learning rate and is approximately constant throughout training. For  $\eta_{\mathrm{crit}} < \eta < \eta_{\mathrm{max}}$  we find that the curvature decreases during training to below  $2 / \eta$ .

Figure 2 also shows the loss initially increasing before converging for large learning rates, a signature of the catapult effect. This transient behavior is very short, taking less than 10 steps to complete. Because of this, the training curve for the test loss is very similar and also shows the catapult effect. In these and other experiments involving ReLU networks, we find that  $\eta_{\mathrm{max}} \approx 12 / \lambda_0$  is a good predictor of the maximum learning rate (in Appendix E.4 we discuss other nonlinearities). We conjecture that this is the typical maximum learning rate of networks with ReLU non-linearities.

# 2.2 GENERALIZATION PERFORMANCE

We now consider the performance of trained models in the different phases discussed in this work. Keskar et al. (2016) observed a correlation between the flatness of a minimum found by SGD and the generalization performance (see Jiang et al. (2020) for additional empirical confirmation of this correlation). In this work, we showed that the minima SGD finds are flatter in the catapult phase, as measured by the top kernel eigenvalue. Our measure of flatness differs from that of Keskar et al. (2016), but we expect that these measures to be correlated.

We therefore conjecture that optimal performance is often obtained for learning rates above  $\eta_{\mathrm{crit}}$  and below the maximum learning rate. In this section we test this conjecture empirically. We find that performance in the large learning rate regime always matches or exceeds the performance when  $\eta < \eta_{\mathrm{crit}}$ . For a fixed compute budget, we find that the best performance is always found in the catapult phase.

Figure 3 shows the performance of a convolutional network and a Wide ResNet (WRN) trained on CIFAR-10. The experimental setup, which we now describe, was chosen to ensure a fair comparison of the performance across different learning rates. The network is trained with different initial learning rates, followed by a decay at a fixed physical time  $t \cdot \eta$  to the same final learning rate. This schedule is introduced in order to ensure that all experiments have the same level of SGD noise toward the end of training.

We present results using two different stopping conditions. In Figure 3a, 3c, all models were trained for a fixed number of training steps. We find a significant performance gap between small and large learning rates, with the optimal learning rate above  $\eta_{\mathrm{crit}}$  and close to  $\eta_{\mathrm{max}}$ . Beyond this learning rate, performance drops sharply.

The fixed compute stopping condition, while of practical interest, biases the results in favor of large learning rates. Indeed, in the limit of small learning rate, training for a fixed number of steps will keep the model close to initialization. To control for this, in Figure 3b,3d models were trained for the same amount of physical time  $t \cdot \eta$ . For the CNN of figure 3b, decaying the learning rate does not have a significant effect on performance and we observe that performance is flat up to  $\eta_{\mathrm{max}}$ , and there is no correlation between our measure of curvature and generalization performance. Figure 3d shows the analogous experiment for WRN. When decaying the learning rate toward the end of training to control for SGD noise, we find that optimal performance is achieved above  $\eta_{\mathrm{crit}}$ . In all these cases,  $\eta_{\mathrm{max}}$  is a good predictor of the maximal learning rate, despite significant differences in the architectures. Notice that by tuning the learning rate to the catapult phase, we are able to achieve performance using MSE loss, and without momentum, that is competitive with the best reported results for this model (Zagoruyko & Komodakis, 2016).

![](images/7847d2cecc505b744eafa06e5638391f4a47e4627c217040f42abf813993f9f1.jpg)  
(a)

![](images/aa0cbfa63ab7a8a0963985f1b8ddc6a4886d5f5b26ccea7128dcf72df98fc7b2.jpg)  
(b)

![](images/5a3c871c61987a2a13e3375778add887bddce9b2fb0bc18d8326a6eca5f3ff26.jpg)  
(c)

![](images/bc12074435aa162c7cf54352da49f3d99ebd5b8a468b7fe733e2cce073b35615.jpg)  
Figure 3: Models perform best with a large learning rate. Test accuracy vs learning rate for (a,b) a CNN trained on CIFAR-10 using SGD with batch size 256 and  $L_{2}$  regularization  $(\eta_{\mathrm{crit}} \approx 10^{-4})$  and (c,d) WRN28-10 trained on CIFAR-10 using SGD with batch size 1024,  $L_{2}$  regularization, and data augmentation  $(\eta_{\mathrm{crit}} \approx 0.14)$ ; see Appendix A for details. (a,c) have a fixed compute budget: (a) 437k steps and (b) 12k steps. (b,d) have been evolved for a fixed amount of physical time: (b) was evolved for  $475 / \eta$  steps (purple) and evolved for 50k more steps at learning rate  $2 \cdot 10^{-5}$  (red) and (d) was evolved for  $3360 / \eta$  steps with learning rate  $\eta$  (purple) and then evolved for 4800 more steps at learning rate 0.035 (red). In all cases, optimal performance is achieved above  $\eta_{\mathrm{crit}}$  and close to the expected maximum learning rate, in agreement with our predictions.  
(d)

In Appendix D.2, we present additional results for WRN on CIFAR-100, with similar conclusions. The fact that optimal performance happens in the catapult phase can also be observed for simple models like a fully-connected ReLU network trained on a subset of MNIST (see the Appendix).

# 3 GRADIENT DESCENT DYNAMICS OF WIDE, 2-LAYER LINEAR NETWORKS

We now turn to a theoretical analysis of the gradient descent dynamics of a two-layer linear network at large but finite width. While such a setting omits complexities such as depth and nonlinearity, our theoretical treatment already reveals the existence of three phases described in Section 1.2 with signatures that match our experiments.

Let the network function  $f$  be given by  $f(x) = n^{-1/2} v^T u x$ . Here  $n$  is the width (number of neurons in the hidden layer),  $u, v \in \mathbb{R}^n$  are the model parameters (collectively denoted  $\theta$ ), and  $x \in \mathbb{R}$  is the training input. At initialization, the weights are drawn from  $\mathcal{N}(0, 1)$ . We prove the following

Theorem 1. Consider a 2-layer linear network of width  $n$ , trained with MSE loss and learning rate  $\eta$ . The training data has a single sample with  $(x, y) = (1, 0)$ . Choose initial values  $f_0 \neq 0$  and  $\lambda_0 > 0$  for the function  $f(x)$  and curvature  $\lambda$ . Let  $\eta_{\mathrm{crit}} \coloneqq 2 / \lambda_0$  and  $\eta_{\mathrm{max}} \coloneqq 4 / \lambda_0$ , and choose  $\delta > 0$ .

1. Lazy phase: If  $\eta < \eta_{\mathrm{crit}}$  then gradient descent achieves loss  $L < \delta$  in  $\mathcal{O}(n^0)$  steps, and the final curvature  $\lambda_f$  obeys  $|\lambda_f - \lambda_0| = \mathcal{O}(n^{-1})$ .  
2. Catapult phase: If  $\eta_{\mathrm{crit}} < \eta < \eta_{\mathrm{max}}$  then gradient descent achieves loss  $L < \delta$ , the final curvature obeys  $\lambda_f \leq 2 / \eta$ , and during optimization the loss grows to be  $\Omega (n\log^{-1}(n))$ .  
3. Divergent phase: If  $\eta \geq \eta_{\mathrm{max}}$  then gradient descent does not converge to a global minimum.

The proof can be found in the appendix. We will complement the theorem with intuition about the dynamics of this model. In the appendix, we also generalize to the case of networks with arbitrary

input dimension. The gradient descent equations at training step  $t$  are

$$
u _ {t + 1} = u _ {t} - \eta n ^ {- 1 / 2} f _ {t} v _ {t}, v _ {t + 1} = v _ {t} - \eta n ^ {- 1 / 2} f _ {t} u _ {t}. \tag {1}
$$

The update equations in function space can be written in terms of the Neural Tangent Kernel. For this model, the kernel evaluated on the training set is a scalar which is equal to  $\lambda$ , its top eigenvalue, and is given by  $\Theta(1,1) = \lambda = n^{-1}\left(\|v\|_2^2 + \|u\|_2^2\right)$ . At initialization, both  $f^2$  and  $\lambda$  scale as  $n^0$  with the width  $n$ . The following update equations for  $f$  and  $\lambda$  at step  $t$  can be derived from equation 1.

$$
f _ {t + 1} = \left(1 - \eta \lambda_ {t} + \frac {\eta^ {2} f _ {t} ^ {2}}{n}\right) f _ {t}, \quad \lambda_ {t + 1} = \lambda_ {t} + \frac {\eta f _ {t} ^ {2}}{n} (\eta \lambda_ {t} - 4). \tag {2}
$$

It is important to note that these are the exact update equations for this model, and that no higher-order terms were neglected. We now analyze these dynamical equations assuming the width  $n$  is large. Two learning rates that will be important in the analysis are  $\eta_{\mathrm{crit}} = 2 / \lambda_0$  and  $\eta_{\mathrm{max}} = 4 / \lambda_0$ .

Lazy phase. Taking the strict infinite width limit, equations equation 2 become

$$
f _ {t + 1} = \left(1 - \eta \lambda_ {t}\right) f _ {t}, \quad \lambda_ {t + 1} = \lambda_ {t}. \tag {3}
$$

When  $\eta < \eta_{\mathrm{crit}}$ ,  $\lambda$  remains constant throughout training. This is a special case of NTK dynamics, where the kernel is constant and the network evolves as a linear model (Lee et al., 2019). The function and the loss both shrink to zero because the multiplicative factor obeys  $|1 - \eta \lambda_t| < 1$ . This convergence happens in  $\mathcal{O}(n^0) = \mathcal{O}(1)$  steps.

**Catapult phase.** When  $\eta_{\mathrm{crit}} < \eta < \eta_{\mathrm{max}}$ , the loss diverges in the infinite width limit. Indeed, from equation 3 we see that the kernel is constant in the limit, while  $f$  receives multiplicative updates where  $|1 - \eta \lambda_t| > 1$ . This is the well-known instability of gradient descent dynamics for linear models with MSE loss. However, the underlying model is not linear in its parameters, and finite width contributions turn out to be important. We therefore relax the infinite width limit and analyze equations (2) for large but finite width,  $n \gg 1$ .

First, note that  $\eta \lambda_0 - 4 < 0$  by assumption, and therefore the (additive) kernel updates are negative for all  $t$ . During early training,  $|f_t|$  grows (as in the infinite width limit) while  $\lambda_t$  remains constant up to small  $\mathcal{O}(n^{-1})$  updates. After  $t \sim \log(n)$  steps,  $|f_t|$  grows to order  $n^{1/2}$ . At this point, the kernel updates are no longer negligible because  $f_t^2/n$  is of order  $n^0$ . The kernel  $\lambda_t$  receives negative, non-negligible updates while both  $f_t$  and the loss continue to grow. This continues until the kernel is sufficiently small that the condition  $\eta \lambda_t \lesssim 2$  is met. We call this curvature-reduction effect the catapult effect. Beyond this point,  $|1 - \eta \lambda_t| < 1$  holds,  $|f_t|$  shrinks, and the loss converges to a global minimum. The  $n$  dependence of the steps until optimization converges is  $\log(n)$ .

It is important for the analysis that we take a modified large width limit, in which the number of training steps grows like  $\log (n)$  as  $n$  becomes large. This is different than the large width limit commonly studied in the literature, in which the number of steps is kept fixed as the width is taken large. When using this modified limit, the analysis above holds even in the limit. Note as well that the catapult effect takes place over  $\log (n)$  steps, and for practical networks will occur within the first 100 steps or so of training.

In the catapult phase, the kernel at the end of training is smaller by an order  $n^0$  amount compared with its value at initialization. The kernel provides a local measure of the loss curvature. Therefore, the minima that SGD finds in the catapult phase are flatter than those it finds in the lazy phase. Contrast this situation, in which the kernel receives non-negligible updates, with the conclusions of Jacot et al. (2018) where the kernel is constant throughout training. The difference is due to the large learning rate, which leads to a breakdown of the linearized approximation even at large width.

Completing the analysis of this model, when  $\eta >\eta_{\mathrm{max}}$  the loss diverges because the kernel receives positive updates, accelerating the rate of growth of the function. Therefore,  $\eta_{\mathrm{max}} = 4 / \lambda_0$  is the maximum learning rate of the model.

# 3.1 NON-PERTURBATIVE PHASE TRANSITION

The large width analysis of the small learning rate phase has been the subject of much work. In this phase, at infinite width, the network map evolves as a linear random features model,  $f_{t+1}^{(0)} = f_t^{(0)} - \Theta f_t^{(0)}$ , where  $f^{(0)}$  is the function of the linearized model. At large but finite width (which we denote by  $n$ ), corrections to this linear evolution can be systematically incorporated via a perturbative expansion (Taylor expansion) around infinite width Dyer & Gur-Ari (2020); Huang & Yau (2019),  $f_t = f_t^{(0)} + \frac{1}{n} f_t^{(1)} + \dots$ . The evolution equations 3 of the solvable model are an example of this. At large width and in the small learning rate phase, the  $O(n^{-1})$  terms are suppressed for all times. In contrast, the leading order dynamics of  $f_t^{(0)}$  diverge when  $\eta > \eta_{\mathrm{crit}}$ , and so the true evolution cannot be described by the linear model. Indeed, the logits grow to  $\mathcal{O}(n^{1/2})$  and thus all terms in equation 3 are of the same order. Similarly, the growth observed empirically in the catapult phase for more general models cannot be described by truncating the perturbative series at any order, because the terms all become comparable.

# 4 DISCUSSION

Previous work (Jacot et al., 2018; Lee et al., 2019; Chizat et al., 2019) has studied the lazy phase of deep neural networks, which is known to occur for sufficiently small learning rates. In this work we argued that the lazy phase exists for learning rates smaller than  $\eta_{\mathrm{crit}} = 2 / \lambda_0$ , where  $\lambda_0$  is the curvature at initialization. This critical learning rate corresponds to where a linear model, constructed from a deep network about its initialized parameters, would diverge under MSE loss. We pointed out the existence of the catapult phase in deep networks, corresponding to the learning rate regime  $\eta_{\mathrm{crit}} < \eta < \eta_{\mathrm{max}}$ . Its unique empirical signatures include the early-time growth of the loss and convergence to a flat minimum. Empirically, the existence and properties of the catapult phase can be observed across a variety network architectures and datasets. At yet larger learning rates  $\eta > \eta_{\mathrm{max}}$ , SGD dynamics are unstable.

A novel analysis illustrating the catapult phase. Through our analytical treatment of a two-layer linear network, we are able to clarify the dynamics behind the catapult effect. Among these, we (i) derived the quantitative changes in loss and curvature and the time scales over which they occur; (ii) derived an expression for  $\eta_{\mathrm{max}}$  in terms of the curvature at initialization; (iii) specified the manner in which the lazy and catapult regimes are distinct phases, which we elaborate on below, in a novel modified infinite-width, infinite-time limit; and (iv) illustrated the dynamical mechanism stabilizing the catapult phase. Our approach reduces to analyzing two coupled difference equations relating the loss and curvature, which we hope may inspire a full treatment of deep networks with nonlinearities.

The change in behavior upon sweeping the learning rate from the lazy to catapult phase is reminiscent of phase transitions that commonly appear in physical systems such as ferromagnets or water, as one changes parameters such as temperature. Indeed, in Appendix C this connection is made concrete, with the change in behavior sharpening as width is increased. In particular, these transitions are non-perturbative: a Taylor series expansion of the linearized model that takes into account finite width corrections is not sufficient to describe the behavior beyond the critical learning rate.

Catapult dynamics often improve generalization. Our results shed light on the regularizing effect of training at large learning rates. The effect presented here is independent of the regularizing effect of stochastic gradient noise, which has been studied extensively. Building on previous works, we noted the observed correlation between flatness and generalization performance. Based on these observations, we expect the optimal performance to often occur for learning rates larger than  $\eta_{\mathrm{crit}}$ , where the linearized model is unstable. Observing this effect required controlling for several confounding factors that affect the comparison of performance between different learning rates. Under a fair comparison, and also for a fixed compute budget, we find that this expectation holds in practice.

One outcome of our work is to address the performance gap between ordinary neural networks, and linear models inspired by the theory of infinite-width networks. Optimal performance is often obtained at large learning rates which are inaccessible to linearized models. In such cases, we expect the performance gap to persist even at arbitrarily large widths. We hope our work can further improve the understanding of deep learning dynamics and performance.

# REFERENCES

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 242-252, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Shun-Ichi Amari, Hyeyoung Park, and Kenji Fukumizu. Adaptive method of realizing natural gradient learning for multilayer perceptrons. Neural computation, 12(6):1399-1409, 2000.  
Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Russ R Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. In Advances in Neural Information Processing Systems, pp. 8139-8148, 2019.  
James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, and Skye Wanderman-Milne. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.  
Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems 32, pp. 2933-2943. Curran Associates, Inc., 2019. URL http://papers.nips.cc/paper/8559-on-lazy-training-in-\differentiable-programming.pdf.  
Amit Daniely. Sgd learns the conjugate kernel class of the network. In Advances in Neural Information Processing Systems, pp. 2422-2430, 2017.  
Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1019-1028. JMLR.org, 2017.  
Simon S. Du, Jason D. Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, pp. 1675-1685, 2019. URL http://proceedings.mlr.press/v97/du19c.html.  
Ethan Dyer and Guy Gur-Ari. Asymptotics of wide networks from feynman diagrams. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1gFvANKDS.  
Jonathan Frankle, David J Schwab, and Ari S Morcos. The early phase of neural network training. arXiv preprint arXiv:2002.10365, 2020.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1-42, 1997.  
Jiaoyang Huang and Horng-Tzer Yau. Dynamics of Deep Neural Networks and Neural Tangent Hierarchy. arXiv e-prints, art. arXiv:1909.08156, Sep 2019.  
Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesà-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 8571-8580. Curran Associates, Inc., 2018.  
Stanislaw Jastrzebski, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho, and Krzysztof Geras. The break-even point on optimization trajectories of deep neural networks. arXiv preprint arXiv:2002.09572, 2020.  
Yiding Jiang, Behnam Neyshabur, Dilip Krishnan, Hossein Mobahi, and Samy Bengio. *Fantastic generalization measures and where to find them.* In *International Conference on Learning Representations*, 2020. URL https://openreview.net/forum?id=SJgIPJBFvH.  
Ryo Karakida, Shotaro Akaho, and Shun-ichi Amari. Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. arXiv e-prints, art. arXiv:1806.01316, June 2018.

Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. CoRR, abs/1609.04836, 2016. URL http://arxiv.org/abs/1609.04836.  
Guillaume Leclerc and Aleksander Madry. The two regimes of deep network training, 2020.  
Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington, and Jascha Sohldickstein. Deep neural networks as gaussian processes. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=B1EA-M-0Z.  
Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d' Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems 32, pp. 8570-8581. Curran Associates, Inc., 2019. URL http://papers.nips.cc/paper/9063-wide-neural-networks-of- \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ (p)  
Yuanzhi Li and Yingyu Liang. Learning overparameterized neural networks via stochastic gradient descent on structured data. In Advances in Neural Information Processing Systems, pp. 8157-8166, 2018.  
Yuanzhi Li, Colin Wei, and Tengyu Ma. Towards explaining the regularization effect of initial large learning rate in training neural networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems 32, pp. 11669-11680. Curran Associates, Inc., 2019.  
Stephan Mandt, Matthew D Hoffman, and David M Blei. Stochastic gradient descent as approximate bayesian inference. The Journal of Machine Learning Research, 18(1):4873-4907, 2017.  
Robert M May. Simple mathematical models with very complicated dynamics. Nature, 261(5560): 459-467, 1976.  
Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. 115(33):E7665-E7671, 2018. doi: 10.1073/pnas.1806579115.  
Naveh, Ben-David, Sompolinsky, and Ringel. to be published.  
Roman Novak, Lechao Xiao, Yasaman Bahri, Jaehoon Lee, Greg Yang, Daniel A. Abolafia, Jeffrey Pennington, and Jascha Sohl-dickstein. Bayesian deep convolutional networks with many channels are gaussian processes. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=B1g30j0qF7.  
Roman Novak, Lechao Xiao, Jiri Hron, Jaehoon Lee, Alexander A. Alemi, Jascha Sohl-Dickstein, and Samuel S. Schoenholz. Neural tangents: Fast and easy infinite neural networks in python. In International Conference on Learning Representations, 2020. URL https://github.com/google/neural-tangents.  
Daniel S. Park, Jascha Sohl-Dickstein, Quoc V. Le, and Samuel L. Smith. The effect of network width on stochastic gradient descent and generalization: an empirical study. CoRR, abs/1905.03776, 2019. URL http://arxiv.org/abs/1905.03776.  
Grant Rotskoff and Eric Vanden-Eijnden. Parameters as interacting particles: long time convergence and asymptotic error scaling of neural networks. In Advances in neural information processing systems, pp. 7146-7155, 2018.  
Justin Sirignano and Konstantinos Spiliopoulos. Mean field analysis of neural networks. arXiv preprint arXiv:1805.01053, 2018.  
Samuel L. Smith and Quoc V. Le. A bayesian perspective on generalization and stochastic gradient descent. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=BBij4yg0Z.

Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, and Quoc V. Le. Don't Decay the Learning Rate, Increase the Batch Size. arXiv e-prints, art. arXiv:1711.00489, Nov 2017.  
Samuel L Smith, Daniel Duckworth, Semon Rezchikov, Quoc V Le, and Jascha Sohl-Dickstein. Stochastic natural gradient descent draws posterior samples in function space. arXiv preprint arXiv:1806.09597, 2018.  
Blake Woodworth, Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Kernel and deep regimes in overparametrized models. arXiv preprint arXiv:1906.05827, 2019.  
Lechao Xiao, Jeffrey Pennington, and Samuel S. Schoenholz. Disentangling trainability and generalization in deep learning, 2019.  
Zeke Xie, Issei Sato, and Masashi Sugiyama. A diffusion theory for deep learning dynamics: Stochastic gradient descent escapes from sharp minima exponentially fast. arXiv preprint arXiv:2002.03495, 2020.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. CoRR, abs/1605.07146, 2016. URL http://arxiv.org/abs/1605.07146.  
Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes over-parameterized deep relu networks. arXiv preprint arXiv:1811.08888, 2018.
