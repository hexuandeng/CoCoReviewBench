# THREE FACTORS INFLUENCING MINIMA IN SGD

Anonymous authors

Paper under double-blind review

# ABSTRACT

We focus on the importance of noise in stochastic gradient descent (SGD) based training of deep neural networks (DNNs). We develop theory that studies SGD training as a stochastic differential equation and show that its stationary distribution is related to the loss surface. Our analysis suggests that the combination of batch size, learning rate, and the variance of the true loss gradients acts as a hyperparameter steering the behavior of SGD and determines the trade-offs between the depth and width of the minima that SGD converges to. In a nutshell, a higher ratio of learning rate to batch size leads to wider minima. We validate our theory by examining the correlation between these three factors and the final performance and sharpness of the minimum found. As a verification of our theory, we empirically demonstrate that the learning dynamics is similar between experiments with different learning rates and batch sizes in SGD if the ratio of learning rate to batch size is the same.

# 1 INTRODUCTION

Despite being massively over-parameterized models (Zhang et al., 2016), deep neural networks (DNNs) have demonstrated good generalization ability and achieved state-of-the-art performances in many application domains such as image (He et al., 2016) and speech recognition (Amodei et al., 2016). The reason for this success has been a focus of research recently but still remains an open question. Our work provides new theoretical insights and useful suggestions for deep learning practitioners.

The standard way of training DNNs corresponds to minimizing a loss function using stochastic gradient descent (SGD) and its variants (Bottou, 1998). In SGD, parameters are updated by taking a small discrete step depending on the learning rate (LR) in the direction of the negative loss gradient, which is approximated based on a small subset of training examples (called mini-batch). Since DNNs are highly non-convex functions with multiple minima, SGD in general converges to different local minima depending on optimization hyper-parameters, initialization and the loss curvature.

Recently, several works (Arpit et al., 2017; Advani & Saxe, 2017; Shirish Keskar et al., 2016) have investigated how SGD impacts generalization in DNNs. It has been argued that local minima with low curvature (wide minima) can generalize better than minima whose curvature is large (sharp minima) (Hochreiter & Schmidhuber, 1997; Shirish Keskar et al., 2016). Specifically, (Shirish Keskar et al., 2016) found that batch size (BS) strongly affects the curvature of minima that are reached by SGD (i.e., smaller batch sizes usually lead to wider minima). We generalize this notion to talk about the controllable noise level in SGD which we theoretically show is determined by the ratio of learning rate over batch size. As such, we discuss how SGD with appropriate noise level tends to converge to deeper and wider minima. In this vein, while (Dinh et al., 2017) discuss the existence of minima with different widths but which behave similarly in terms of predictions, we argue that SGD naturally tends to find wider minima when appropriate noise levels are used.

We approximate SGD as a continuous stochastic differential equation (Bottou, 1998; Mandt et al., 2017). Based on this, we derive the stationary distribution of this stochastic process, and further derive the relative probability of landing in one local minima as compared to another depending on their depth and width. Our main finding is that the ratio of learning rate and batch-size along with noise about the true gradient (due to the choice of batch sample) influences the trade-off between the depth and sharpness of the final minima, with a high ratio of learning rate to batch size favouring flatter minima. In addition, our analysis provides a theoretical justification for the emper

ical observation that scaling the learning rate linearly with batch size leads to good performance of DNNs (Krizhevsky, 2014; He et al., 2016).

We verify our theoretical insights experimentally on several models and datasets. In particular, we demonstrate that high learning rate to batch size ratio (due to either high learning rate or low batch-size) leads to wider minima and correlates well with better validation performance. We also observe that a high learning rate to batch size ratio helps to prevent memorization. Furthermore, we show that multiplying each of the learning rate and the batch size by the same scalar factor results in similar training dynamics, but if the scalar factor gets too large, then the assumptions of the theory are violated and performance drops.

# 2 RELATED WORK

The relationship between stochastic gradient descent (SGD) and sampling a posterior distribution via stochastic Langevin methods has been the subject of discussion in a number of papers (Chen et al., 2014; Ding et al., 2014; Vollmer et al., 2015; Welling & Teh, 2011; Shang et al., 2015; Sato & Nakagawa, 2014). In particular, Mandt et al. (2017) describe the dynamics of stochastic gradient descent (SGD) as a stochastic process that can be divided into three distinct phases. In the first phase, weights diffuse and move away from the initialization. In the second phase the gradient magnitude dominates the noise in the gradient estimate. In the final phase, the weights are near the optimum. (Shwartz-Ziv & Tishby, 2017) make related observations from an information theoretic point of view and suggest the diffusion behavior of the parameters in the last phase leads to the minimization of mutual information between the input and hidden representation. In a similar vein, we relate the SGD dynamics to the stationary distribution of the stochastic differential equation. Our derivation bears similarity with Mandt et al. (2017). However, while Mandt et al. (2017) study SGD as an approximate Bayesian inference method in the final phase of optimization in a locally convex setting, our end goal is to analyse the stationary distribution over the entire parameter space reached by SGD. Further, our analysis allows us to compare the probability of SGD ending up in one minima over another, which is novel in our case.

Our work is closely related to the ongoing discussion about the role of large batch size and sharpness of found minima in generalization (Shirish Keskar et al., 2016). Goyal et al. (2017); Hoffer et al. (2017) empirically show that scaling up the learning rate, and training for more epochs, leads to good generalization using large batch size. Our novelty is in explaining from the theoretical point of view the importance of the specific ratio of learning rate to batch size. We also observe that the final minima found by large batch size can be wide if a larger learning rate is used.

Our work is also related to the importance of noise in SGD, which has been previously explored. The main inspiration behind learning rate schedule has been shown to be noise annealing (Bottou, 1998). Neelakantan et al. (2015) observe empirically that adding noise can aid optimization of very deep networks. Our analysis allows us to derive the impact of the gradient's noise in the SGD stationary distribution. Additionally, our work also provides intuitions toward explaining the recently proposed Cyclic learning rate (CLR) schedule (Smith, 2015). Cyclic learning rate schedules have demonstrated good optimization and generalization performances, but are grounded on empirical observation. We also show that one can replace learning rate annealing with an equivalent batch size schedule. It suggests that the benefit of cyclic learning rate relates to the noise that it induces.

# 3 THEORETICAL RESULTS

Our focus in this section is on finding the relative probability with which we end optimization in a certain minimum, as compared to another minimum. We will find that the relative probability depends on the local geometry of the loss function at each minimum, and on the BS, the LR and the covariance of the stochastic gradients.

# 3.1 SETUP

We follow a theoretical setup similar to Mandt et al. (2017), approximating SGD with a continuous-time stochastic process, which we now outline.

Let us consider a model parametrized by  $\pmb{\theta} = \{\theta_1, \dots, \theta_q\}$ . For  $N$  training examples  $\pmb{x}_i, i \in \{1, \dots, N\}$ , the loss function,  $L(\pmb{\theta})$ , and the corresponding gradient  $\mathbf{g}(\pmb{\theta})$ , are defined based on the sum over the loss values for all training examples. Stochastic gradients  $\mathbf{g}^{(S)}(\pmb{\theta})$  arise when we consider a batch  $\mathcal{B}$  of size  $S < N$  of random indices drawn uniformly from  $\{1, \dots, N\}$  and form an (unbiased) estimate of loss and gradient based on the corresponding subset of training examples

$$
L ^ {(S)} (\boldsymbol {\theta}) = \frac {1}{S} \sum_ {n \in \mathcal {B}} l (\boldsymbol {\theta}, \boldsymbol {x} _ {n}), \quad \mathbf {g} ^ {(S)} (\boldsymbol {\theta}) = \frac {\partial}{\partial \boldsymbol {\theta}} L ^ {(S)} (\boldsymbol {\theta}).
$$

We consider stochastic gradient descent (SGD) with learning rate  $\eta$ , as defined by the update rule

$$
\boldsymbol {\theta} (t + 1) = \boldsymbol {\theta} (t) - \eta \boldsymbol {g} ^ {(S)} (\boldsymbol {\theta}).
$$

We now make the following assumptions:

(1) By the central limit theorem (CLT), we assume the gradient noise is Gaussian with covariance matrix  $\frac{1}{S}\mathbf{C}(\boldsymbol{\theta})$

$$
\mathbf {g} ^ {(S)} (\boldsymbol {\theta}) = \mathbf {g} (\boldsymbol {\theta}) + \frac {1}{\sqrt {S}} \Delta \mathbf {g} (\boldsymbol {\theta}), \text {w h e r e} \Delta \mathbf {g} (\boldsymbol {\theta}) \sim N (0, \mathbf {C} (\boldsymbol {\theta})) .
$$

We note that the covariance is symmetric positive-semidefinite, and so can be decomposed into the product of two matrices  $\mathbf{C}(\pmb{\theta}) = \mathbf{B}(\pmb{\theta})\mathbf{B}^{\top}(\pmb{\theta})$ .

(2) We assume the discrete process of SGD can be approximated by the continuous time limit of the following stochastic differential equation (known as a Langevin equation)

$$
\frac {d \boldsymbol {\theta}}{d t} = - \eta \mathbf {g} (\boldsymbol {\theta}) + \frac {\eta}{\sqrt {S}} \mathbf {B} (\boldsymbol {\theta}) \mathbf {f} (t) \tag {1}
$$

where  $\mathbf{f}(t)$  is a normalized Gaussian time-dependent stochastic term.

# 3.2 MAIN RESULTS

The Langevin equation is a stochastic differential equation, and we are interested in its stationary distribution to gain insights into the behavior of SGD and the quality of minima it converges to. It can be derived by finding the stationary solutions of a partial differential equation known as the Fokker-Planck equation, which governs the evolution of the probability density of the value of the parameters with time<sup>1</sup>. The Fokker-Planck equation and its derivation can be found in Appendix A in equation (5). We arrive at the following theorem (all the notations are defined in the setup section above):

Theorem 1 (Stationary Distribution). Assume that the gradient covariance is proportional to the identity, i.e.  $\mathbf{C}(\pmb{\theta}) = \sigma^2(\pmb{\theta})\mathbf{I}$  and that  $|\nabla_{\pmb{\theta}}(\sigma^2(\pmb{\theta}))| \ll 2S|\pmb{g}|\eta$ . Then the stationary distribution of the stochastic differential equation 6 is given by

$$
P (\boldsymbol {\theta}) = P _ {0} e ^ {- L (\boldsymbol {\theta}) / n}, \tag {2}
$$

where  $n \equiv \frac{\sigma^2(\theta)\eta}{2S}$  and  $P_0$  is a normalization constant.

Here  $P(\theta)$  defines the density over the parameter space. The above result says that if we run SGD long enough (under the assumptions made), then the likelihood of the parameters being in a particular state asymptotically follows the above density. Note, that  $n \equiv \frac{\sigma^2(\theta)\eta}{2S}$  is a measure of the noise in the system and it depends on two parts:  $n_c \equiv \eta / S$ , which is the controllable noise set by the choice of hyper-parameters, and the gradient variance  $\sigma^2(\theta)$ , which is not a tunable hyper-parameter.

Given the probability density  $P(\theta)$ , we are now interested in deriving the probability of ending at a given minimum,  $\hat{\theta}_A$ , which we will denote by lowercase  $p_A = \tilde{p}_AC$ , where  $C$  is a normalization constant which is the same for every mimnima (the unnormalized probability  $\tilde{p}_A$  is all we are interested in when estimating the relative probability of finishing in a given minimum compared to another one). This probability is derived in Appendix D, and given in the following theorem, which is the core result of our theory.

Theorem 2 (Probability of ending in minima A). Assume the loss is locally convex with Hessian  $\mathbf{H}_A$  and loss  $L_{A}$  at the minimum  $\theta_{A}$ . Then the unnormalized probability of ending in minima  $\theta_{A}$  is

$$
\tilde {p} _ {A} = \sqrt {\frac {(2 \pi n _ {A}) ^ {q}}{\operatorname* {d e t} \mathbf {H} _ {A}}} \exp \left(- \frac {L _ {A}}{n _ {A}}\right) \tag {3}
$$

where  $q$  is the number of parameters and  $n_A \equiv n_c \sigma^2 (\pmb{\theta}_A)$  is the noise at the minimum.

Here we see that the probability of landing in a given minimum depends on the geometry of the loss surface (captured by the height  $L_{A}$  and the determinant of the Hessian  $\operatorname{det} \mathbf{H}_A$ , which determines the width of the minimum) as well as on the noise  $n_{A}$  at  $\theta_{A}$ .

For larger noise,  $n_A$ , the factor  $\exp \left(-\frac{L_A}{n_A}\right)$  is down-weighted compared to  $\sqrt{\frac{(2\pi n_A)^q}{\operatorname*{det} \mathbf{H}_A}}$ , which grows with larger  $n_A$ . Thus, with larger noise the width gets more important than the height of the minimum, while the converse holds if the noise is low enough. Also notice, that with  $n_A$  the probability  $\tilde{p}_A$  directly depends on the ratio of learning rate and batch size  $n_c = \eta / S$ . We will explore the implications of this experimentally in Section 4.

To see which kind of minima is preferred, it is instructive to consider the ratio of probabilities  $p_A$  and  $p_B$  given by

$$
\frac {p _ {A}}{p _ {B}} = \sqrt {\frac {n _ {A} ^ {p}}{n _ {B} ^ {p}}} \sqrt {\frac {\det \mathbf {H} _ {B}}{\det \mathbf {H} _ {A}}} \exp \left(\frac {L _ {B}}{n _ {B}} - \frac {L _ {A}}{n _ {A}}\right).
$$

Let us now look at the special case where  $n_A = n = n_B$  (other cases can be found in Appendix E). Without loss of generality we can set  $L_{A} \geq L_{B}$ . If  $L_{A} = L_{B}$  the minimum with lower determinant of the Hessian (i.e. the flatter minima) is more probable. If  $L_{A} \geq L_{B}$ , it holds that  $p_A \geq p_B$  if and only if

$$
\frac {1}{n} \leq \frac {\log \left(\sqrt {\frac {\det \mathbf {H} _ {B}}{\det \mathbf {H} _ {A}}}\right)}{(L _ {A} - L _ {B})} \equiv Y.
$$

That is, there is an upper bound on the inverse of the noise for  $\theta_{A}$  to be favored in the case that its loss is higher than at  $\theta_{B}$ , and this upper bound depends on the difference in the heights compared to the ratio of the widths. In particular we can see that if  $\operatorname{det} \mathbf{H}_B < \operatorname{det} \mathbf{H}_A$ , then  $Y < 0$ , and so no amount of noise will result in  $\theta_{A}$  being more probable than  $\theta_{B}$  - that is, if the minimum at  $\theta_{A}$  is both higher and sharper than the minimum at  $\theta_{B}$ , it is never reached with higher probability than  $\theta_{B}$ , regardless of the amount of noise. However, if  $\operatorname{det} \mathbf{H}_B > \operatorname{det} \mathbf{H}_A$  then  $Y > 0$ , and there is a lower bound on the noise

$$
n > \frac {\left(L _ {A} - L _ {B}\right)}{\log \left(\sqrt {\frac {\operatorname* {d e t} \mathbf {H} _ {B}}{\operatorname* {d e t} \mathbf {H} _ {A}}}\right)} \tag {4}
$$

to make  $\theta_{A}$  more probable than  $\theta_{B}-$  that is if the minimum at  $\theta_{A}$  is higher but flatter than the minimum at  $\theta_{B}$ , it is favored over  $\theta_{B}$ , as long as the noise is large enough, as defined by eq. (4).

To summarize, the presented theory shows that the noise level in SGD (which is defined by the learning rate, batch size and gradient covariance) controls the extent to which optimization favors wider over lower minima. Increasing the noise by increasing the ratio of learning rate and batch size, increases the probability of wider compared to lower optima.

# 4 EXPERIMENTS

# 4.1 IMPACT OF SGD ON MINIMA

In this section, we empirically study the impact of the learning rate  $\eta$ , batch size  $S$  on the local minimum that SGD finds.

We first focus on an MLP with 20 layers with ReLU activation functions (similar to one used in (Neelakantan et al., 2015)) and trained on FashionMNIST. We study how the controllable noise

![](images/51a1c8b0bfb9031bc2a53296eefc0c127f1677689730d04465760efdef45f448.jpg)  
(a) Correlation of  $\frac{\eta}{S}$  with logarithm of norm of Hessian.

![](images/40ad91c92e63f37c2523211640f2e20b1a14d8ff4d6c5e94b3e6fccebfcf7087.jpg)  
(b) Correlation of  $\frac{\eta}{S}$  with validation accuracy.

![](images/9c6a27c134cf1bfe40e129e1c1452dac4031ae28bdbd21a42e542c8baba99481.jpg)  
Figure 1: Impact on SGD with ratio of learning rate (LR)  $\eta$  and batch size (BS)  $S$  for 20 layer ReLU Network on FashionMNIST.  
(a) left  $\frac{\eta = 0.1}{S = 128}$ , right  $\frac{\eta = 0.1}{S = 1024}$  
Figure 2: Interpolation of Resnet56 Networks trained with different with different learning rate to batch size ratio,  $\frac{\eta}{N}$ .  $\alpha$  (x-axis) corresponds to the interpolation coefficient. Lower  $\frac{\eta}{N}$  ratio leads to sharper minima.

![](images/6f8e8af458f543a01fc3348493a9906da96a74956395b7d5b5d57222972b7267.jpg)  
(b) left  $\frac{\eta = 0.1}{S = 128}$ , right  $\frac{\eta = 0.01}{S = 128}$

![](images/89c0f2c9cb0bb9894b1d2486acc8b92804d8cf64fad8e20a4f5fd5207a3838c7.jpg)  
(c) left  $\frac{\eta = 0.1}{S = 1024}$ , right  $\frac{\eta = 0.01}{S = 128}$

ratio  $n_c = \frac{\eta}{S}$  leads to minima with different curvatures and validation accuracy. To measure the curvature at that minimum, we compute the norm of its Hessian using a finite difference (Wu et al., 2017) (higher Hessian norm implies higher sharpness of minima). In Figure 1a, we report the norm of the Hessian for local minima obtained by SGD for different  $n_c$  with  $\frac{\eta}{S}$  (where  $\eta \in [5e - 3, 5e - 2]$  and  $S \in [50, 1000]$ ). As  $n_c = \frac{\eta}{S}$  grows, we observe that the norm of the Hessian at the minima also decreases, suggesting that higher  $\frac{\eta}{S}$  pushes the optimization towards a flatter minimum. This agrees with Theorem 2, (3), that higher  $\frac{\eta}{S}$  favors flatter over sharper minima.

In Figure 1b, we explore the impact of  $n_c = \frac{\eta}{S}$  on the final validation performance and confirm that better generalization correlates with higher  $n_c$ . Taken together, Fig. 1a and Fig. 1b imply wider minima correlate well with better generalization. As  $n_c = \frac{\eta}{S}$  increases, SGD finds local minima that generalize better. In the Appendix, we report similar results for Resnet applied on CIFAR10 and the 20 layer ReLU network with bad initialization.

To further illustrate the behavior of SGD with different noise levels, we train three Resnet56 models on CIFAR10 using naive SGD, but with different  $\frac{\eta}{S}$ . Our baseline model uses  $\frac{\eta=0.1}{S=128}$ . We define a large batch model  $\frac{\eta=0.1}{S=1024}$  and a small learning rate model  $\frac{\eta=0.01}{S=128}$ . The large batch and the small learning rate model have approximately the same  $\frac{\eta}{S}$  ratio. In Figure 2, we follow (Goodfellow et al., 2014) and look at the loss between interpolation of different pairs of models. More specifically, let  $f_{1}$  and  $f_{2}$  be the early stop models of two SGD optimizers using different  $\frac{\eta}{S}$ . In Fig. 2, we report  $L((1-\alpha)f_{1} + \alpha f_{2}$  where  $L$  is the loss function and  $\alpha \in [-1,2]$ .

We observe that both model with large batch or low learning rate end in a sharper minimum relatively to the baseline model, both having a lower  $\frac{\eta}{S}$  than the baseline. Each of the three plots in Figure 2 adds empirical weight to our theoretical prediction that higher  $n_c = \eta / S$  gives preference to wider minima over sharper minima.

Finally we perform a similar experiment on VGG-11 on CIFAR-10, but in this case, we train all the models with the same noise level but different values of learning rate and batch size. Specifically, in this case we use  $\frac{\eta = 0.1 \times \beta}{S = 50 \times \beta}$ , where we use  $\beta = 0.25, 1, 4$ . We then interpolate between the model

![](images/9491c8bc4c05987bf3c0af135b000680e0cc2ddb67468efdcff215bcccd6e14d.jpg)  
(a)  $\beta = 1$  corresponds to model at  $\alpha = 0$  and  $\beta = 4$  (b)  $\beta = 1$  corresponds to model at  $\alpha = 0$  and  $\beta = 4$  corresponds to model at  $\alpha = 1$  corresponds to model at  $\alpha = 0.25$

![](images/bf8b46a4666690cdac25c3fd46bc9ab58e06f21e9bb9da483efb72e83292c08b.jpg)

![](images/a2c684069f6bca0a1c394e269cb3964143bb9af27880949ba66f7c1ad414ea7e.jpg)  
Figure 3: Left:  $\frac{\eta = 0.1 \times \beta}{S = 50 \times \beta}$ , right:  $\frac{\eta = 0.1 \times \beta}{S = 50 \times \beta}$ . Interpolation between parameters of models trained with the same learning rate ( $\eta$ ) to batch-size ( $S$ ) ratio, but different  $\eta$  and  $S$  values determined by  $\beta$ . Hence our theory predicts the minima for these models should be qualitatively similar as can be seen by these plots.

![](images/0e62e3e724c6ac1e3d7d5877c5c2f39954d4eaca3800e22dd7360d721af49892.jpg)  
Figure 4: Learning rate schedule can be replaced by an equivalent batch size schedule. The ratio of learning rate to batch size is equal at all times in both red and blue in each plot. Train and test accuracy for experiments involving VGG-11 architecture on CIFAR10 dataset. Left: cyclic batch size schedule (blue) in range 128 to 640, compared to cyclic learning rate schedule (red) in range 0.001 to 0.005. Right: constant batch size 128 and constant learning rate 0.001 (blue), compared to constant batch size 640 and constant learning rate 0.005 (red).

parameters of the pair of models and record loss and accuracy (shown in figure 3). We find that all the minima in this case have similar width and depth qualitatively showing that for the same noise ratio, SGD ends up in minima of similar quality.

# 4.2  $\frac{\eta}{S}$  RATIO DETERMINES LEARNING DYNAMICS OF SGD

A prediction of the theory is that the learning rate  $\eta$  and batch size  $S$  only appear in the ratio  $n_c = \eta / S$  in the expressions for the probability of finishing in a given minimum. Based on this, the endpoint of SGD with a learning rate schedule  $\eta \rightarrow \eta / a$ , for some  $a > 0$ , and a constant batch size  $S$ , should be the same as the endpoint of SGD with a constant learning rate and a batch size schedule  $S \rightarrow aS$ . We see in Fig. 4 that exchanging a learning rate schedule for a batch size schedule leads to similar train and test accuracy, for both cyclic and constant schedules, in the case where the controllable noise  $n_c = \eta / S$  is the same in each case. In particular we notice very similar dynamics in the case of a cyclic learning rate when compared to a cyclic batch size. Full loss curves and learning rate and batch size schedules for this experiment can be found in Appendix G.2 in Fig. 10. The test accuracy for cyclic batch size and cyclic learning rate was  $89.39\%$  and  $89.24\%$  respectively. For constant (batch size, learning rate) = (128, 0.001) and constant (batch size, learning rate) = (640, 0.005) the test accuracy was  $87.25\%$  and  $86.92\%$  respectively.

Note that exchanging batch size for learning rate such that the ratio  $\eta / S$  remains constant is different to the common assumption that one should trade them so as to keep the ratio  $\eta / \sqrt{S}$  constant. This commonly used heuristic of scaling the learning rate with the square root of the batch size is used for example in (Hoffer et al., 2017), as a way of keeping the covariance matrix of the parameter update step the same for any batch size. However, our point here is that our theory and experiments

![](images/db102dc359371eafd08c55708ef27070c8ed8bd70beeac7d05e4943d6d790459.jpg)  
Figure 5:  $\frac{\eta}{S}$  impact on memorization on dataset with added  $25\%$  and  $50\%$  random label noise on the training set. The two left columns (and right columns) use 0.9 momentum (and 0.0 momentum) respectively. We observe that given a specific level of memorization, high  $\frac{\eta}{S}$  leads to better generalization.

![](images/885cbafe7765f3f2c2aeb13bedf47cc13b69b623dc4dc0b89f707324386e126a.jpg)

![](images/0ea3f61109ff6f30c383fecd5c28c1424b7d37fdc83e2c3fcc6335648d35fd29.jpg)

![](images/c66099a5a2c5d4b991bc4f09da6817cfb5b254440f439b8563aeee6b8a97e42c.jpg)

suggest exchanging learning rate and batch size in a way that keeps the ratio of controllable noise  $n_c = \eta / S$  constant will result in the same stationary distribution.

# 4.3 IMPACT OF SGD ON MEMORIZATION

To generalize well, a model must identify the underlying pattern in the data instead of simply perfectly memorizing each training example. An empirical approach to test for memorization is to see if a deep neural network can fit a training set with random noise labels rather than the true labels (Zhang et al., 2016; Arpit et al., 2017). In this section, we highlight that SGD with a sufficient amount of noise reduces the amount of memorization in a network.

Experiments are performed using the MNIST dataset and an MLP similar to the one in (Arpit et al., 2017), but with 256 hidden units. We train the MLP on the dataset with different amounts of random labels in the training set. For each label noise level, we evaluate the impact of  $\frac{\eta}{S}$ , controlling the amount of noise in the SGD and its impact on the generalization performances. Specifically, we run a grid of batch size in 50, 100, 200, 400, 800, learning rate in 0.05, 0.1, 0.2 and momentum in 0.0 and 0.9. Models are trained for 300 epochs. We report in Fig. 5 the MLP performances for both the noisy training set and the clean validation set (without random label, to measure the model generalization ability).

Our analysis highlights that SGD with low controllable noise  $n_c = \frac{\eta}{S}$  steers the endpoint of optimization towards a minima with low generalization ability, associated with a sharp minima, as predicted by our theory. We observe in Fig. 5 that larger noise in SGD (regardless if controlled using smaller batch size or larger learning rate) leads to solutions which generalize better for the same amount of memorized random labels on the training set. We also reproduce the observation reported in (Arpit et al., 2017): that randomization roughly starts after reaching maximum generalization - for more details see the curves in Fig. 11 included in Appendix G.3. They show that lower controllable noise  $\frac{\eta}{S}$  optimize much slower and do not reach larger than  $20\%$  accuracy on random labels. For runs with momentum we exclude higher learning rates than 0.02 as they lead to divergence (see appendix).

# 4.4 BREAKING POINT OF THE THEORY IN PRACTICE

Our analysis relies on the fact that the gradient step is sufficiently small so that first order approximation of a Taylor expansion is a good estimate of the loss function. In the case where the learning rate becomes too high, the first order approximation of a Taylor expansion is no longer suitable, in which case the continuous limit of the discrete SGD update equation will no longer be valid. In this case, the stochastic differential equation doesn't hold, and hence neither does the Fokker-Planck equation, and so our theory will not be predictive. In particular, we don't expect to arrive at the same stationary distribution as governed just by the ratio  $\eta / S$ . This is exemplified in Fig 6, where similar learning dynamics and final performance is seen when simultaneously multiplying the learning rate and batch size by a factor  $\beta$  up to a certain limit. This is done for different train-set size to see if the breaking point depends on this size. This plots seem to suggest that this happens for smaller  $\beta$

![](images/abbb7b30381713b0345e8128d84ab78d2721138b7365d0568b2d664cff32d9f5.jpg)  
(a) Train dataset size 12000

![](images/9f9e1f20162f49134b858ff73d2a247558693a2653ce990ffde164983f902642.jpg)  
Figure 6: Breaking point of the theory: Experiments involving VGG-11 architecture on CIFAR10 dataset. Validation accuracy for different dataset sizes, and different  $\beta$  values. In each experiment, we multiply the learning rate  $(\eta)$  and batch size  $(S)$  with  $\beta$  such that the ratio  $\frac{\eta \times \beta}{S \times \beta}$  is fixed. We observe that for the same ratio increasing the learning rate and batch size yields similar performance up to a  $\beta$  and then performance drops significantly.

![](images/a2f7b43383a181e834f66883c51696f74b104fee8b5f8e532acb6377d88bd9b3.jpg)  
(b) Train dataset size 22500  
(c) Train dataset size 45000

values when dataset size is smaller. We highlight other limitations due to our theory assumption in appendix F. A similar experiment is performed on Resnets in figure 7.

# 4.5 CYCLICAL BATCH AND LEARNING RATE SCHEDULES

It has been observed that a cyclic learning rate (CLR) schedule leads to better generalization (Smith, 2015). In Sec. 4.2 we demonstrated that one can exchange cyclic learning rate schedule (CLR) with batch size (CBS) and approximately preserve the practical benefit of CLR. This inspired us to hypothesize that by changing between controllable noise levels  $\left(\frac{\eta}{S}\right)$  CLR switches between sharp/deep and wide/shallow minima. To validate that, we run VGG-11 on CIFAR10 using 4 training schedules: CLR with stepsize of 4 and 15 epochs in each stage, CBS with stepsize 4 and 15 epochs in each stage. Each run is repeated 8 times and for each variant we track evolution of sharpness and accuracy. We observe that CBS and CLR with longer stages lead to  $89.9 \pm 0.20\%$  and  $90.2 \pm 0.10\%$  test accuracy, in both cases  $0.2 - 0.3\%$  above variance with shorter range. CBS and CLR with long stepsize each seem to be promising schedules for training DNNs. Finally, we validate that CBS and CLR switch between sharp/deep and wide/shallow minima, suggesting that CLR improves convergence time to stationary distribution, as seen in Fig. 13 of Appendix G.4.

# 5 CONCLUSIONS

We shed light on the role of noise in SGD optimization of DNNs and argued that three factors (batch size, learning rate and gradient variance) strongly influence the properties (loss and width) of the final minima at which SGD converges. Learning rate and batch size of SGD can be viewed as one effective hyper-parameter that acts as a factor controllable noise  $n_c = \eta / S$ , which, together with the gradient covariance influences the trade-off between the loss and width of the final minima (high noise favors wide minima), which in turn tunes the robustness of the prediction function.

Further, we experimentally verify that the controllable noise  $n_c = \eta / S$  determines the width and height of the minima towards which SGD converges. We also show the impact of this controllable noise on the memorization phenomenon. We discussed the limitations of the theory and in what situations it breaks down, exemplified by when the learning rate gets too large. We also experimentally verify that  $\eta$  and  $S$  can vary in linear proportion as long as the controllable noise  $\eta / S$  remains the same. In addition, our experiments suggest that cyclical learning rates oscillate between sharp/deep and wide/shallow minima as long as the stage of increased noise is long enough to allow for mixing.

# REFERENCES

Madhu S Advani and Andrew M Saxe. High-dimensional dynamics of generalization error in neural networks. arXiv preprint arXiv:1710.03667, 2017.

Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International Conference on Machine Learning, pp. 173-182, 2016.  
Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 233-242, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR. URL http://proceedings.mlr.press/v70/arpit17a.html.  
Léon Bottou. Online learning and stochastic approximations. On-line learning in neural networks, 17(9):142, 1998.  
T. Chen, E. B. Fox, and C. Guestrin. Stochastic gradient Hamiltonian Monte Carlo. In Proceedings of the 31st International Conference on Machine Learning, pp. 1683-1691, 2014.  
N. Ding, Y. Fang, R. Babbush, C. Chen, R. D. Skeel, and H. Neven. Bayesian sampling using stochastic gradient thermostats. In Advances in Neural Information Processing Systems 27, pp. 3203-3211, 2014.  
L. Dinh, R. Pascanu, S. Bengio, and Y. Bengio. Sharp Minima Can Generalize For Deep Nets. ArXiv e-prints, March 2017.  
Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization problems. arXiv preprint arXiv:1412.6544, 2014.  
P. Goyal, P. Dollar, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. ArXiv e-prints, June 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1-42, 1997.  
E. Hoffer, I. Hubara, and D. Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. ArXiv e-prints, May 2017.  
Alex Krizhevsky. One weird trick for parallelizing convolutional neural networks. arXiv preprint arXiv:1404.5997, 2014.  
S. Mandt, M. D. Hoffman, and D. M. Blei. Stochastic Gradient Descent as Approximate Bayesian Inference. *ArXiv e-prints*, April 2017.  
A. Neelakantan, L. Vilnis, Q. V. Le, I. Sutskever, L. Kaiser, K. Kurach, and J. Martens. Adding Gradient Noise Improves Learning for Very Deep Networks. ArXiv e-prints, November 2015.  
Issei Sato and Hiroshi Nakagawa. Approximation analysis of stochastic gradient Langevin dynamics by using fokker-planck equation and its process. In Eric P. Xing and Tony Jebara (eds.), Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pp. 982-990, Beijing, China, 22-24 Jun 2014. PMLR. URL http://proceedings.mlr.press/v32/satoa14.html.  
Xiaocheng Shang, Zhanxing Zhu, Benedict Leimkuhler, and Amos J Storkey. Covariance-controlled adaptive Langevin thermostat for large-scale Bayesian sampling. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett (eds.), Advances in Neural Information Processing Systems 28, pp. 37-45, 2015. URL http://papers.nips.cc/paper/5978-covariance-controlled-adaptive-langevin-thermostat-for-large-scale-bayesian-pdf.

N. Shirish Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang. On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. ArXiv e-prints, September 2016.  
Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. CoRR, abs/1703.00810, 2017. URL http://arxiv.org/abs/1703.00810.  
L. N. Smith. Cyclical Learning Rates for Training Neural Networks. ArXiv e-prints, June 2015.  
S. J. Vollmer, K. C. Zygalakis, and Y. W. Teh. (Non-) asymptotic properties of stochastic gradient Langevin dynamics. arXiv preprint arXiv:1501.00438, 2015.  
M. Welling and Y. W. Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning, pp. 681-688, 2011.  
Lei Wu, Zhanxing Zhu, et al. Towards understanding generalization of deep learning: Perspective of loss landscapes. arXiv preprint arXiv:1706.10239, 2017.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.
