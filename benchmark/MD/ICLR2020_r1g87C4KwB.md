# THE BREAK-EVEN POINT ON THE OPTIMIZATION TRAJECTORIES OF DEEP NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Understanding the optimization trajectory is critical to understand training of deep neural networks. We show how the hyperparameters of stochastic gradient descent influence the covariance of the gradients  $(\mathbf{K})$  and the Hessian of the training loss  $(\mathbf{H})$  along this trajectory. Based on a theoretical model, we predict that using a high learning rate or a small batch size in the early phase of training leads SGD to regions of the parameter space with (1) reduced spectral norm of  $\mathbf{K}$ , and (2) improved conditioning of  $\mathbf{K}$  and  $\mathbf{H}$ . We show that the point on the trajectory after which these effects hold, which we refer to as the break-even point, is reached typically early during training. We demonstrate these effects empirically for a range of deep neural networks applied to multiple different tasks. Finally, we apply our analysis to networks with batch normalization (BN) layers and find that it is necessary to use a high learning rate to achieve loss smoothing effects attributed previously to BN alone.

# 1 INTRODUCTION

The choice of the optimization method implicitly regularizes deep neural networks (DNNs) by influencing the optimization trajectory in the loss surface (Neyshabur, 2017; Arora, 2019). In this work, we theoretically and empirically investigate how the learning rate and the batch size used at the beginning of training determine properties of the entire optimization trajectory.

![](images/edec28da3b128fb4a3a2bc175dc417f7ec3142d06dce6cf806dbc2e3cf268130.jpg)  
Figure 1: Left: Early part of the training trajectory on CIFAR-10 (before reaching  $65\%$  training accuracy) of a simple CNN model optimized using SGD with learning rate  $\eta = 0.1$  (red) and  $\eta = 0.01$  (blue). Each point (model) on the trajectory is represented by its test predictions embedded into a two-dimensional space using UMAP. The background color indicates the spectral norm of  $\mathbf{K}$  (brighter is higher). Depending on  $\eta$ , after reaching what we call the break-even point, trajectories are steered towards regions characterized by different  $\mathbf{K}$ . Right: The spectral norm of  $\mathbf{K}$  along the trajectory for  $\eta = 0.01$  against the distances to the closest point from the trajectory for  $\eta = 0.1$  (y axis). Vertical line marks the highest spectral norm of  $\mathbf{K}$  achieved along the trajectory for  $\eta = 0.1$ .

![](images/4d7796cadc6f6efc04b6eb4e1392fbc5eb1c79dfa0c4d7833e915211dad8ce8a.jpg)

We focus our analysis on two objects that quantify different properties of the optimization trajectory: the Hessian of the training loss  $(\mathbf{H})$ , and the covariance of gradients  $(\mathbf{K})$ . The matrix  $\mathbf{K}$  quantifies noise induced by noisy estimate of the full-batch gradient and has been also linked to generalization error (Roux et al., 2008; Fort et al., 2019). The matrix  $\mathbf{H}$  describes the curvature of the loss surface and is often connected to optimization speed. Further, better conditioning of  $\mathbf{H}$  and  $\mathbf{K}$  has been attributed as the main reason behind the efficacy of batch normalization (Bjorck et al., 2018; Ghorbani et al., 2019).

Our first and main contribution is predicting and empirically demonstrating two effects induced in the early phase of training by the choice of the hyperparameters in stochastic gradient descent (SGD): (1) reduced spectral norms of  $\mathbf{K}$  and  $\mathbf{H}$  and (2) improved conditioning of  $\mathbf{K}$  and  $\mathbf{H}$ . These effects manifest themselves after a certain point on the optimization trajectory, to which we refer to as the break-even point. See Fig. 1 for an illustration of this phenomenon. We make our predictions based on a theoretical model of the initial phase of training, which incorporates recent observations on the instability and oscillations in the parameter space that characterize the learning dynamics of neural networks (Masters & Luschi, 2018; Xing et al., 2018; Lan et al., 2019).

As our second contribution, we apply our analysis to a network with batch normalization (BN) layers and find that our predictions are valid in this case too. Delving deeper in this direction of investigation, we show that using a large learning rate is necessary to reach well-conditioned regions of the loss surface, which was previously attributed to BN alone (BJORCK et al., 2018; Ghorbani et al., 2019; Page, 2019).

# 2 RELATED WORK

Learning dynamics and the early phase of training. Our theoretical model is motivated by recent work on the learning dynamics of neural networks (Goodfellow et al., 2014; Masters & Luschi, 2018; Wu et al., 2018; Yao et al., 2018; Xing et al., 2018; Jastrzebski et al., 2018; Lan et al., 2019). We are most directly inspired by Xing et al. (2018); Jastrzebski et al. (2018); Lan et al. (2019) who show that training oscillates in the parameter space, and by Wu et al. (2018) who proposes a linear stability approach to studying how SGD selects a minimum.

In our work we argue that the initial phase of training has important implications for the rest of the trajectory. This is directly related to Erhan et al. (2010); Achille et al. (2017) who propose the existence of the critical period of learning. Erhan et al. (2010) argue that initial training, unless pre-training is used, is sensitive to shuffling of examples in the first epochs of training. Achille et al. (2017); Golatkar et al. (2019); Sagun et al. (2017); Keskar et al. (2017) demonstrate that adding regularizers in the beginning of training affects the final generalization disproportionately more compared to doing so later.

The covariance of the gradients and the Hessian. The covariance of the gradients, which we denote by  $\mathbf{K}$ , encapsulates the geometry and magnitude of variation in gradients across different samples (Thomas et al., 2019). The matrix  $\mathbf{K}$  was related to the generalization error in Roux et al. (2008). A similar quantity, cosine alignment between gradients computed on individual examples, was recently shown to explain some aspects of deep networks generalization (Fort et al., 2019).

The second object that we study is the Hessian that quantifies the loss surface shape (LeCun et al., 2012). Recent work has shown that the largest eigenvalues of  $\mathbf{H}$  grow quickly initially, and then stabilize at a value dependent on the learning rate and the batch size (Keskar et al., 2017; Sagun et al., 2017; Fort & Scherlis, 2019; Jastrzebski et al., 2018). The Hessian can be decomposed into a sum of two terms, where the dominant term (at least at the end of training) is the uncentered covariance of gradients  $\mathbf{G}$  (Sagun et al., 2017; Papyan, 2019). While we study  $\mathbf{K}$ , the centered version of  $\mathbf{G}$ ,  $\mathbf{K}$  and  $\mathbf{G}$  are typically similar due to the dominance of noise in training (Zhu et al., 2018; Thomas et al., 2019).

Implicit regularization induced by optimization method. Multiple prior work study the regularization effects that are attributed only to the optimization method (Neyshabur, 2017). A popular

research direction is to bound the generalization error based on the properties of the final minimum such as the norm of the parameter vector or the Hessian (Bartlett et al., 2017; Keskar et al., 2017). Perhaps the most related work to ours is (Arora et al., 2019; Arora, 2019). They suggest it is necessary to study the trajectory to understand generalization of deep networks. In this vein, but in contrast to most of the previous work, we focus (1) on the implicit regularization effects that can be attributed to the GD dynamics at the beginning of training, and (2) on the covariance of gradients.

# 3 TWO CONJECTURES ABOUT SGD TRAJECTORY

In this section we make two conjectures about the optimization trajectory induced by SGD based on a theoretical model of the learning dynamics in the early stage of training.

**Definitions.** Let us denote loss on an example  $(\mathbf{x},y)$  by  $\mathcal{L}(\mathbf{x},y;\theta)$ , where  $\theta$  is a  $D$ -dimensional parameter vector. A key object we study is the Hessian  $\mathbf{H}$  of the training loss. The second key object we study is the covariance of the gradients  $\mathbf{K} = \frac{1}{N}\sum_{i=1}^{N}(g_i - g)^T(g_i - g)$ , where  $g_i = g(\mathbf{x}_i, y_i; \theta)$  is the gradient of  $\mathcal{L}$  with respect to  $\theta$  calculated on  $i$ -th example,  $N$  is the number of training examples, and  $g$  is the full-batch gradient. We denote the  $i$ -th normalized eigenvector and eigenvalue of a matrix  $\mathbf{A}$  by  $e_A^i$  and  $\lambda_A^i$ . Both  $\mathbf{H}$  and  $\mathbf{K}$  are computed at a given  $\theta$ , but we omit this dependence in the notation. Let  $t$  index steps of optimization, and  $\theta(t)$  the parameter vector at optimization step  $t$ .

Inspired by Wu et al. (2018) we study stability of optimization, in our case restricted to  $e_H^1$ . Let us call the projection of parameters  $\theta$  onto  $e_H^1$  by  $\psi = \langle \theta, e_H^1 \rangle$ . With a slight abuse of notation let  $g(\psi) = \langle g(\theta), e_H^1 \rangle$ . Similarly to Wu et al. (2018), we say SGD is unstable along  $e_H^1$  at  $\theta(t)$  if the norm of elements of sequence  $\psi(\tau + 1) = \psi(\tau) - \eta g(\psi(\tau))$  diverges when  $\tau \to \infty$ , where  $\psi(0) = \theta(t)$ . The sequence  $\psi(\tau)$  represents optimization trajectory restricted from  $\theta(t)$  only to the direction of  $e_H^1$ .

Assumptions. We make the following assumptions to build our model:

1. The loss surface for each example projected onto  $e_H^1$  is a quadratic function. This assumption is also used by Wu et al. (2018). It was shown to hold for the loss averaged over examples by Alain et al. (2019). It is also well known that the spectral norm of  $\mathbf{H}$  is positive (Sagun et al., 2017).  
2. The eigenvectors  $e_H^1$  and  $e_K^1$  are co-linear, i.e.  $e_H^1 = \pm e_K^1$ , and furthermore  $\lambda_K^1 = \alpha \lambda_H^1$  for some  $\alpha \in \mathbb{R}$ . This is inspired by Papyan (2019) who show that  $\mathbf{H}$  can be approximated by  $\mathbf{G}$  (uncentered  $\mathbf{K}$ ).  
3. When in a region that is not stable along  $e_H^1$ , training trajectory steers towards more stable regions by decreasing  $\lambda_H^1$ . This is inspired by recent work showing training can escape region with too large curvature compared to the learning rate (Zhu et al., 2018; Wu et al., 2018; Jastrzebski et al., 2018).  
4. The spectral norm of  $\mathbf{H}$ ,  $\lambda_1^H$ , increases during training, unless increasing  $\lambda_1^H$  would lead to entering a region where training is not stable along  $e_H^1$ . This is inspired by (Keskar et al., 2017; Sagun et al., 2017; Jastrzebski et al., 2018; Fort & Scherlis, 2019) who show that in many settings  $\lambda_1^H$  increases in the beginning of training.

These assumptions are only used to build a theoretical model for the early phase of training. Its main purpose is to make predictions about the training procedure that we verify empirically.

Reaching the break-even point earlier for a larger learning rate or a smaller batch size. Let us restrict to the case when training is initialized at  $\theta(0)$  at which SGD is stable along  $e_H^1(0)$ . We aim to show that the learning rate  $(\eta)$  and the batch size  $(S)$  determine  $\mathbf{H}$  and  $\mathbf{K}$  in our model, and conjecture that the same holds in real neural networks.

Consider two optimization trajectories for  $\eta_{1}$  and  $\eta_{2}$ , where  $\eta_{1} > \eta_{2}$ , that are initialized at the same  $\theta_{0}$ , where optimization is stable along  $e_H^1 (t)$  and  $\lambda_H^1 (t) > 0$ . Under Assumption 1 the loss surface

along  $e_H^1 (t)$  can be expressed as  $f(\psi) = \sum_{i = 1}^{N}(\psi -\psi^{*})^{2}H_{i}(t)$ , where  $H_{i}(t)\in \mathbb{R}$ . It can be shown that at any iteration  $t$  the necessary and sufficient condition for SGD to be stable along  $e_H^1 (t)$  is:

$$
(1 - \eta \lambda_ {1} ^ {H} (t)) ^ {2} + s (t) ^ {2} \frac {\eta^ {2} (N - S)}{S (N - 1)} \leq 1, \tag {1}
$$

where  $N$  is the training set size and  $s(t)^2 = \mathrm{Var}[H_i(t)]$  over the training examples. A proof can be found in (Wu et al., 2018). We call this point on the trajectory on which the LHS of Eq. 1 becomes equal to 1 for the first time the break-even point.

Under the Assumption 3,  $\lambda_1^H(t)$  and  $\lambda_1^K(t)$  increase over time. If  $s = N$ , the break-even point is reached at  $\lambda_1^H(t) = \frac{2}{\eta}$ . More generally, it can be shown that for  $\eta_1$ , the break-even point is reached for a lower magnitude of  $\lambda_1^H(t)$  than for  $\eta_2$ . The same reasoning can be carried out for  $S$ . We state this formally and prove in App. A.

From this point on the trajectory, under Assumption 4, SGD does not enter regions where either  $\lambda_1^H(t')$  or  $\lambda_1^K(t')$  is larger than at the break-even point, as otherwise it would lead to increasing one of the terms in LHS of Eq. 1, and hence losing stability along  $e_H^1(t')$ .

The two conjectures about real DNNs. Assuming that real DNNs reach the break-even point, we make the following two conjectures, arising from our theoretical model above, about their optimization trajectory. The most direct implication of reaching the break-even point is that  $\lambda_{K}^{1}$  and  $\lambda_{H}^{1}$  at the break-even point depend on  $\eta$  and  $S$ , which we formalize as:

Conjecture 1 (Variance reduction effect of SGD). Along the SGD trajectory, the maximum attained values of  $\lambda_H^1$  and  $\lambda_K^1$  are smaller for a larger learning rate or a smaller batch size.

We refer to Con. 1 as variance reduction effect of SGD, because reducing  $\lambda_K^1$  can be shown to reduce the  $L_{2}$  distance between the full-batch gradient, and the mini-batch gradient.

Next, we make another, stronger, conjecture. It is plausible to assume that reaching the break-even point does not affect the  $\lambda_H^i$  and  $\lambda_K^i$  for  $i\neq 1$ , because increasing their values does not impact stability along  $e_H^1$  in our theoretical model. It is also well known that a large number of eigenvalues of  $\mathbf{H}$  increase initially (Fort & Scherlis, 2019; Sagun et al., 2017). Based on these remarks we conjecture that:

Conjecture 2 (Pre-conditioning effect of SGD). Along the SGD trajectory, the maximum attained values of  $\frac{\lambda_K^*}{\lambda_K^1}$  and  $\frac{\lambda_H^*}{\lambda_H^1}$  are larger for a larger learning rate or a smaller batch size, where  $\lambda_K^*$  and  $\lambda_H^*$  are the smallest nonzero eigenvalues of  $\mathbf{H}$  and  $\mathbf{K}$ , respectively. Furthermore, the maximum attained values of  $\mathrm{Tr}(\mathbf{K})$ ,  $\mathrm{Tr}(\mathbf{H})$  are smaller for a larger learning rate or a smaller batch size.

We consider non-zero eigenvalues in the conjecture, because  $K$  has  $N - D$  non-zero eigenvalues, where  $N$  is the number of training points. In practice we will measure  $\mathrm{Tr}(\mathbf{K}) / \lambda_K^1$ , which is an upper-bound on  $\lambda_K^* / \lambda_K^1$ .

# 4 EXPERIMENTS

In this section we first analyse learning dynamics in the early phase of training. Next, we empirically validate the two conjectures. In the final part we extend our analysis to a neural network with batch normalization layers.

Due to the space constraint we take the following approach to reporting results. In the main body of the paper, we focus on the CIFAR-10 dataset (Krizhevsky, 2009) and the IMDB dataset (Maas et al., 2011), to which we apply three architectures: a vanilla CNN (SimpleCNN) following Keras example (Chollet et al., 2015), ResNet-32 (He et al., 2015a), and LSTM (Hochreiter & Schmidhuber, 1997). We also validate the two conjectures for DenseNet (Huang et al., 2016) on the ImageNet (Deng et al., 2009) dataset, BERT (Devlin et al., 2018b) fine-tuned on the MNLI dataset (Williams et al., 2017), and a multi-layer perceptron on the FashionMNIST dataset (Xiao et al., 2017). These results are in the Appendix. We include all experimental details in App. C.

Following Dauphin et al. (2014); Alain et al. (2019), we estimate the top eigenvalues and eigenvectors of  $\mathbf{H}$  using the Lanczos algorithm on a random subset of  $5\%$  of the training set on CIFAR-10.

![](images/32edcd51c58882e01a58c6a61833c98f3f1197836815e64ac6aabb7c76d95dee.jpg)  
Figure 2: The evolution of  $\lambda_{K}^{1}$  (spectral norm of  $\mathbf{K}$ ),  $\lambda_{H}^{1}$  (spectral norm of  $\mathbf{H}$ ), and  $\alpha^{*}$  (width of the loss surface, see text for details) in the early phase of training. Consistently with our theoretical model,  $\lambda_{K}^{1}$  is correlated initially with  $\lambda_{H}^{1}$  (left) and  $\alpha^{1}$  (right). The training reaches a smaller maximum value of  $\lambda_{K}^{1}$  and  $\lambda_{H}^{1}$  for a higher learning rate.

![](images/2140d454311e9b82613e671fee8a50c448128b00fb83f00f933e3de206e6cdc8.jpg)

We estimate the top eigenvalues and eigenvectors of  $\mathbf{K}$  using (in most cases) batch size of 128 and approximately  $5\%$  of the training set on CIFAR-10. We describe the procedure in more details, as well as compare to using batch size of 1, in App. B.

# 4.1 A CLOSER LOOK AT THE EARLY PHASE OF TRAINING

In this section we examine the learning dynamics in the early phase of training. Our goal is to verify some of the assumptions made in Sec. 3. We analyse the evolution of  $\lambda_H^1$  and  $\lambda_K^1$  for  $\eta = 0.02$  and  $\eta = 0.2$  using the SimpleCNN on the CIFAR-10 dataset.

Are  $\lambda_{K}^{1}$  and  $\lambda_{H}^{1}$  correlated in the beginning of training? The key assumption behind our theoretical model is that  $\lambda_{K}^{1}$  and  $\lambda_{H}^{1}$  are correlated, at least prior to reaching the break-even point. We confirm this in Fig. 2. The highest achieved  $\lambda_{K}^{1}$  and  $\lambda_{H}^{1}$  are larger for the smaller  $\eta$ . Additionally, we observe that after achieving the highest value of  $\lambda_{H}^{1}$ , further growth of  $\lambda_{K}^{1}$  does not translate into an increase of  $\lambda_{K}^{1}$ . This is expected:  $\lambda_{H}^{1}$  decays to 0 when the mean loss decays to 0 for cross entropy loss (Martens, 2016).

Does training become increasingly unstable in the early phase of training? A key aspect of our model is that an increase of  $\lambda_K^1$  and  $\lambda_H^1$  translates into a decrease in stability, which we formalized as stability along  $e_H^1$ . Computing stability directly along  $e_H^1$  is computationally expensive. Instead, we measure a more tractable proxy. Let us define  $\alpha^*$  to be the minimal increment of the SGD step length such that the training loss increases by at least  $20\%$ . More precisely, let  $\theta(t)$  and  $\theta(t+1)$  denote the two consecutive steps on the optimization trajectory. We define  $\alpha^*$  as the minimum  $\alpha$  such that  $\mathcal{L}(\theta(t) - \alpha^*(\theta(t) - \theta(t+1))) \geq 1.2\mathcal{L}(\theta(t))$ . In Fig. 2 we observe that  $\alpha^*$  is anti-correlated with  $\lambda_K^1$ , which is consistent with our theoretical model. We also observe that  $\alpha^*$  reaches a value close to 2 for both tested  $\eta$ , which suggests reaching the break-even point.

Visualizing the break-even point. Finally, to understand the break-even point phenomenon better, we visualize the learning dynamics leading to reaching the break-even point in our model in Fig. 1 (left). Following Erhan et al. (2010), we embed the test set predictions at each step of training of SimpleCNN, in our case using UMAP (McInnes et al., 2018). In the Figure we observe that the trajectory corresponding to  $\eta = 0.01$  diverges from the trajectory corresponding to  $\eta = 0.1$  when entering a region of the loss surface characterized by a high  $\lambda_{K}^{1}$ . Because a low dimensional embedding can mischaracterize the true distances (Wattenberg et al., 2016), we confirm our interpretation by plotting the  $L_{2}$  distance to the closest iteration of  $\eta = 0.1$  trajectory in the right panel of Fig. 1.

Summary. We have shown that the dynamics of the early phase of training is consistent with the assumptions made in our model. That is,  $\lambda_K^1$  and  $\lambda_H^1$  increase approximately proportionally to each other, which is also correlated with a decrease of a proxy of stability. Finally, we have shown qualitatively reaching the break-even point.

# 4.2 VARIANCE REDUCTION AND PRE-CONDITIONING EFFECT OF SGD

In this section we validate empirically Con. 1 and Con. 2 in three settings. For each model we pick manually a suitable range of learning rates and batch sizes to ensure that the properties of  $\mathbf{K}$  and  $\mathbf{H}$  that we examine have converged in a reasonable computational budget; we use 200 epochs on CIFAR-10 and 50 epochs on IMDB.

We summarize the results in Fig. 3, Fig. 4 for SimpleCNN, ResNet-32, and LSTM. Curves are smoothed with a moving average. The training curves, as well as experiments for other architectures and datasets (including a DenseNet on ImageNet and BERT on MNLI) can be found in App. D.

![](images/54567c113d5f09642b20acb9964a6305941323fd09f0771c5c25e7983865d014.jpg)

![](images/a43cc32f53aef7ced150b947f112ab5c1479abf0db9a1e53087b8ef36ff3b34d.jpg)  
(a) SimpleCNN on the CIFAR-10 dataset

![](images/9f8871cf501a021cb3b48db50faf0bf2ea0ca09a1a3772570e06fede20c78ccd.jpg)

![](images/dbc3f38feb13471884c2c035abbf76d772bcb239a07e8a3fe2293d206aff222a.jpg)

![](images/a00a7c465064a38d43b1d45e8ffabce82115d72ab98a08ca71305fe6dedee5d3.jpg)

![](images/54a2ea265ec07861c0f64952b70d2bc499bd26377f89c3a1a3062031e95b5fa6.jpg)  
(b) ResNet-32 on the CIFAR-10 dataset

![](images/ec30be89cb16ced510e2af434cba45f6839cb1532e92e31395eaa601ce8235ef.jpg)

![](images/f64e269717fb8deef5bd293aa5ddbf00b55f8a84647cfe6960863eb12864b988.jpg)

![](images/cee9ff7761c3ce5a014db86c2275216956ab79267cab10dd417806cd2ebea5e0.jpg)  
Figure 3: The variance reduction and the pre-conditioning effect of SGD in various settings. Trajectories corresponding to higher learning rates  $(\eta)$  or lower batch sizes  $(S)$  are characterized by lower maximum  $\lambda_{K}^{1}$  (variance reduction) and larger maximum  $\mathrm{Tr}(\mathbf{K}) / \lambda_{K}^{1}$  (better conditioning). These effects occur early in training. Vertical lines mark the first epoch at which training accuracy is above  $60\%$  for CIFAR-10, and above for  $75\%$  for IMDB.

![](images/3b8d762c6c15442c334465d4d1f68f07f75a1e20612c93b9e683244dba359848.jpg)  
(c) LSTM on the IMDB dataset

![](images/3e2f486a0a21a3e29af4bf271ef551830ba9f58deb178d6e1162081129946b3f.jpg)

![](images/776f06e06e951b9a96b17ee579d9c822bfc77eb32611b9cad75ff6eb5a746d5d.jpg)

Null hypothesis. A natural assumption is that the choice of  $\eta$  or  $S$  does not influence  $\mathbf{K}$  and  $\mathbf{H}$  along the optimization trajectory. In particular, it is not self-evident that using a high  $\eta$ , or a small  $S$ , would steer optimization towards better conditioned regions of the loss surface.

Conjecture 1. To validate Conjecture 1 we examine the highest value of  $\lambda_K^1$  observed along the optimization trajectory. As visible in Fig. 3 using a higher  $\eta$  results in  $\lambda_K^1$  achieving a lower maximum. For instance  $\max (\lambda_K^1) = 0.87$  and  $\max (\lambda_K^1) = 3.01$  for  $\eta = 0.1$  and  $\eta = 0.01$ , respectively. Similarly, we can conclude that using a higher  $S$  in SGD leads to reaching a higher value of  $\lambda_K^1$ .

Recall that we compute  $\lambda_K^1$  using a constant batch size of 128. While we say that low  $S$  leads to variance reduction (lower maximum  $\lambda_K^1$ ), this is not contradictory to the fact that increasing  $S$  generally decreases variance of mini-batch gradients.

Conjecture 2. To test Conjecture 2 we compute the maximum value of  $\mathrm{Tr}(\mathbf{K}) / \lambda_K^1$  along the optimization trajectory. It is visible in Fig. 3 that using a higher  $\eta$  results in a lower minimum value of  $\mathrm{Tr}(\mathbf{K}) / \lambda_K^1$ . For instance,  $\max (\mathrm{Tr}(\mathbf{K}) / \lambda_K^1) = 14.29$  and  $\max (\mathrm{Tr}(\mathbf{K}) / \lambda_K^1) = 10.69$  for  $\eta = 0.1$  and  $\eta = 0.01$ , respectively. Similarly, we can conclude from these plots that using a higher  $S$  leads to lower  $\max (\mathrm{Tr}(\mathbf{K}) / \lambda_K^1)$ .

Due to space constraints we move Figures showing the effect of  $\eta$  and  $S$  on  $\mathrm{Tr}(\mathbf{K})$  to App. 3. We observe that the maximum of  $\mathrm{Tr}(\mathbf{K})$  depends on  $\eta$  and  $S$  in the same way as  $\lambda_K^1$ .

How early in training is the break-even point reached? How  $\lambda_H^1$  and  $\lambda_K^1$  depend on  $\eta$  and  $S$  at the end of training was already studied by Jastrzebski et al. (2017); Keskar et al. (2017); Wu et al. (2018). Importantly, we find that  $\lambda_K^1$  and  $\lambda_H^1$  reach the highest value early in training: close to reaching  $60\%$  training accuracy on CIFAR-10, and  $75\%$  training accuracy on IMDB. See also the vertical lines in Fig. 3.

Other experiments. We report how  $\lambda_H^1$  depend on  $\eta$  and  $S$  for ResNet-32 and SimpleCNN in Fig. 4. We observe that the conclusions carry over to  $\lambda_H^1$ , which is consistent with experiments in Jastrzebski et al. (2018). We found the effect on  $\mathrm{Tr}(\mathbf{H}) / \lambda_H^1$  of  $\eta$  and  $S$  to be weaker. This might be because, in contrast to  $\mathrm{Tr}(\mathbf{K})$ , we approximate  $\mathrm{Tr}(\mathbf{H})$  using only the top five eigenvalues (see App. B for details).

Summary In this section we have demonstrated the variance reduction (Conjecture 1) and the preconditioning effect (Conjecture 2) of SGD. Furthermore, we have shown these effects occur early in training. We also found that conclusions carry over to other settings including BERT on MNLI and DenseNet on ImageNet (see App. D).

![](images/cd6e61821b66618d0daea81c35eca73727c24409728ad0d2368c5d6725a7eecb.jpg)  
(a) ResNet-32

![](images/160e08775b695f188946ef76bd85a195b89f150a068b32f71b97f2041f5fcfcb.jpg)  
Figure 4: The variance reduction of SGD, for ResNet-32 (left) and SimpleCNN (right). Trajectories corresponding to higher learning rates ( $\eta$ ) or smaller batch size ( $S$ ) are characterized by a lower maximum  $\lambda_H^1$ . Vertical lines mark the first epoch at which training accuracy is above  $60\%$ .

![](images/ef70d6b9d391a833e98afd54e4b66279900b8b1d141fc2314921298b877c3de3.jpg)  
(b) SimpleCNN

![](images/49198ac2556897f53be0f1d80a5ae82509ea16ef26b72099f23d0346bede1d60.jpg)

# 4.3 IMPORTANCE OF  $\eta$  FOR CONDITIONING IN BATCH NORMALIZED NETWORKS

![](images/5e3fa91ccc7e509139a060e044d33918f0e58f2a714709a6f19c81b61ba8d15a.jpg)  
(a) Left:  $\frac{\|g\|}{\|g_5\|}$  for SimpleCNN-BN and SimpleCNN. Right:  $\lambda_H^1$  and  $\lambda_K^1$  for SimpleCNN-BN

![](images/12fb0c9d4c3491e2df95ee80e975655f15f6e1d16c66c23151e7972ca0a8c599.jpg)

![](images/b4ebaf0fa404294c9acd413aa7050394c01ba47e4079f8413136b2e481c19d63.jpg)

![](images/f034f10cb53438d19fb915e7c7b393003f4af020e6149da75599a4cfef421369.jpg)

![](images/547209b51e68fc7631835d9e5b2fdf41f9a33e86d844a52059f933be36c70c4c.jpg)  
(b) Left:  $\| \gamma \|$  of the last BN layer. Middle:  $\lambda_K$ . Right:  $\frac{\operatorname{Tr}(\mathbf{K})}{\lambda_K^1}$  for SimpleCNN-BN

![](images/81499701666a2fcf11518ba7e8b3ab9c6382cb0e77bff22f502cfbe215433067.jpg)  
Figure 5: Evolution of various metrics that quantify conditioning of the loss surface for SimpleCNN with and without batch normalization layers (SimpleCNN and SimpleCNN-BN), and for different learning rates.

![](images/be73e4982b2d3e0011ad11db848df8ed9e059d86e981f110dde161c0b78eb3da.jpg)

The loss surface of deep networks has been widely reported to be ill-conditioned, which is the key motivation behind using second order methods in deep learning (LeCun et al., 2012; Martens &

Grosse, 2015). Recently, Ghorbani et al. (2019); Page (2019) have argued that the key reason behind the efficacy of batch normalization (Ioffe & Szegedy, 2015) is improving conditioning of the loss surface. Our Conjecture 2 is that using a high  $\eta$  or a low  $S$  results as well in improved conditioning. A natural question that we investigate in this section is how the two phenomena are related.

Are the two conjectures valid in batch normalized networks? First, to investigate whether our conjectures hold in batch normalized network, we run similar experiments as in Sec. 4.2 on a SimpleCNN model with batch normalization layers inserted after each convolutional layer (SimpleCNN-BN), using the CIFAR-10 dataset. We test  $\eta \in \{0.001, 0.01, 0.1, 1.0\}$ ;  $\eta = 1.0$  leads to divergence of SimpleCNN without BN. We summarize the results in Fig. 5. The evolution of  $\mathrm{Tr}(\mathbf{K}) / \lambda_K^1$  and  $\lambda_K^1$  show that both Conjecture 1 and Conjecture 2 hold in this setting.

A closer look at the early phase of training. To further corroborate that our analysis applies to BN networks, we study the early phase of training of a network with batch normalization layers, complementing the results in Sec. 4.1.

We observe in Fig. 5 (bottom) that training of SimpleCNN-BN starts in a region characterized by relatively high  $\lambda_K^1$ . This is consistent with prior work showing that batch normalized networks lead to gradient explosion in the first iteration (Yang et al., 2019).  $\lambda_K^1$  then decays for all but the lowest  $\eta$ . This behavior is consistent with our theoretical model. We also track the norm of the scaling factor in BN,  $\| \gamma \|$ , in the last layer of the network in Fig. 5 (bottom). It is visible that  $\eta = 1.0$  and  $\eta = 0.1$  initially decrease the value of  $\| \gamma \|$ , which we hypothesize to be one of the mechanisms by which high  $\eta$  steers optimization towards better conditioned regions of the loss surface in BN networks.

BN requires using a high learning rate. As our conjectures hold for BN network, a natural question is if learning can be ill-conditioned with a low learning rate even when BN is used. Ghorbani et al. (2019) show that without BN, mini-batch gradients are largely contained in the subspace spanned by the top eigenvectors of noncentered  $\mathbf{K}$ . To answer this question we track  $\| g\| /\| g_5\|$ , where  $g$  denotes the mini-batch gradient, and  $g_{5}$  denotes the mini-batch gradient projected onto the top 5 eigenvectors of  $\mathbf{K}$ . A value of  $\| g\| /\| g_5\|$  close to 1 implies that the mini-batch gradient is mostly contained in the subspace spanned by the top 5 eigenvectors of  $\mathbf{K}$ .

We compare two settings: (1) SimpleCNN-BN optimized with  $\eta = 0.001$ , and (2) SimpleCNN optimized with  $\eta = 0.01$ . We make three observations. First, the maximum (minimum) value of  $\| g \| / \| g_5 \|$  is 1.90 (1.37) and 1.88 (1.12), respectively. Second, the maximum value of  $\lambda_K^1$  is 10.3 and 16, respectively. Finally,  $\mathrm{Tr}(\mathbf{K}) / \lambda_K^1$  reaches 12.14 in the first setting, and 11.55 in the second setting. Comparing these differences to differences that are induced by using the highest  $\eta = 1.0$  in SimpleCNN-BN, we can conclude that using a large learning rate is necessary to observe the effect of loss smoothing which was previously attributed to BN alone (Ghorbani et al., 2019; Page, 2019; Bjorck et al., 2018). This might be directly related to the result that a high learning rate is necessary to achieve good generalization when using BN (Bjorck et al., 2018).

Summary. We have shown that our analysis applies to a network with batch normalization layers, and that using a high learning rate is necessary in a batch normalized network to improve conditioning of the loss surface relatively to the same network without batch normalization.

# 5 CONCLUSION

Based on a theoretical model, we conjectured and empirically argued for the existence of the break-even point on the optimization trajectory induced by SGD. Next, we demonstrated that using a high learning rate or a small batch size in SGD has two effects on  $\mathbf{K}$  and  $\mathbf{H}$  along the trajectory that we referred to as (1) variance reduction and (2) pre-conditioning.

There are many potential implications of the existence of the break-even point. We investigated one in particular, and demonstrated that using a high learning rate is necessary to achieve the loss smoothing effects previously attributed to batch normalization alone.

Additionally, the break-even occurs typically early during training, which might be related to the recently discovered phenomenon of the critical learning period in training of deep networks (Achille et al., 2017; Golatkar et al., 2019). We plan to investigate this connection in the future.

# REFERENCES

Alessandro Achille, Matteo Rovere, and Stefano Soatto. Critical learning periods in deep neural networks. CoRR, abs/1711.08856, 2017.  
Guillaume Alain, Nicolas Le Roux, and Pierre-Antoine Manzagol. Negative eigenvalues of the hessian in deep neural networks. CoRR, abs/1902.02366, 2019.  
Sanjeev Arora. Is optimization a sufficient language for understanding deep learning? 2019.  
Sanjeev Arora, Nadav Cohen, Noah Golowich, and Wei Hu. A convergence analysis of gradient descent for deep linear neural networks. In International Conference on Learning Representations, 2019.  
Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30. Curran Associates, Inc., 2017.  
Nils Bjorck, Carla P Gomes, Bart Selman, and Kilian Q Weinberger. Understanding batch normalization. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31. Curran Associates, Inc., 2018.  
François Chollet et al. Keras, 2015.  
Yann N. Dauphin, Razvan Pascanu, Caglar Güçehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. CoRR, abs/1406.2572, 2014.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018a.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018b.  
Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and Samy Bengio. Why does unsupervised pre-training help deep learning? J. Mach. Learn. Res., 11, March 2010. ISSN 1532-4435.  
Stanislav Fort and Adam Scherlis. The goldilocks zone: Towards better understanding of neural network loss landscapes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, 2019.  
Stanislav Fort, Pawe Krzysztof Nowak, and Srini Narayanan. Stiffness: A new perspective on generalization in neural networks, 2019.  
Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Time Matters in Regularizing Deep Networks: Weight Decay and Data Augmentation Affect Early Learning Dynamics, Matter Little Near Convergence. arXiv e-prints, art. arXiv:1905.13277, May 2019.  
Ian J. Goodfellow, Oriol Vinyals, and Andrew M. Saxe. Qualitatively characterizing neural network optimization problems. arXiv e-prints, art. arXiv:1412.6544, Dec 2014.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015a.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015b.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural Comput.*, 9(8), November 1997. ISSN 0899-7667.  
Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks. CoRR, abs/1608.06993, 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32Nd International Conference on International Conference on Machine Learning - Volume 37, ICML'15. JMLR.org, 2015.  
Stanislaw Jastrzebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos J. Storkey. Three factors influencing minima in SGD. CoRR, abs/1711.04623, 2017.  
Stanislaw Jastrzebski, Zachary Kenton, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. On the Relation Between the Sharpest Directions of DNN Loss and the SGD Step Length. arXiv e-prints, art. arXiv:1807.05031, Jul 2018.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.  
Janice Lan, Rosanne Liu, Hattie Zhou, and Jason Yosinski. LCA: Loss Change Allocation for Neural Network Training. arXiv e-prints, art. arXiv:1909.01440, Sep 2019.  
Yann A. LeCun, Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller. Efficient Back-Prop, pp. 9-48. Springer Berlin Heidelberg, Berlin, Heidelberg, 2012. ISBN 978-3-642-35289-8. doi: 10.1007/978-3-642-35289-8_3. URL https://doi.org/10.1007/978-3-642-35289-8_3.  
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, Portland, Oregon, USA, June 2011. Association for Computational Linguistics.  
James Martens. Second-order optimization for neural networks. University of Toronto (Canada), 2016.  
James Martens and Roger B. Grosse. Optimizing neural networks with kronecker-factored approximate curvature. CoRR, abs/1503.05671, 2015.  
Dominic Masters and Carlo Luschi. Revisiting small batch training for deep neural networks. CoRR, abs/1804.07612, 2018.  
Leland McInnes, John Healy, Nathaniel Saul, and Lukas Grossberger. Umap: Uniform manifold approximation and projection. The Journal of Open Source Software, 3(29), 2018.  
Behnam Neyshabur. Implicit regularization in deep learning. CoRR, abs/1709.01953, 2017.  
David Page. How to train your resnet 7: Batch norm. 2019.  
Vardan Papyan. Measurements of three-level hierarchical structure in the outliers in the spectrum of deepnet hESSians. CoRR, abs/1901.08244, 2019.  
Nicolas L. Roux, Pierre antoine Manzagol, and Yoshua Bengio. Topmoumoute online natural gradient algorithm. In J. C. Platt, D. Koller, Y. Singer, and S. T. Roweis (eds.), Advances in Neural Information Processing Systems 20. Curran Associates, Inc., 2008.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3), 2015.  
Levent Sagun, Utku Evci, V. Ugur Güney, Yann Dauphin, and Léon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. CoRR, abs/1706.04454, 2017.  
Valentin Thomas, Fabian Pedregosa, Bart van Merrienboer, Pierre-Antoine Manzagol, Yoshua Bengio, and Nicolas Le Roux. Information matrices and generalization. CoRR, abs/1906.07774, 2019.  
Martin Wattenberg, Fernanda Vivas, and Ian Johnson. How to use t-sne effectively. Distill, 2016.  
Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. CoRR, abs/1704.05426, 2017.  
Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers). Association for Computational Linguistics, 2018.  
Lei Wu, Chao Ma, and Weinan E. How sgd selects the global minima in over-parameterized learning: A dynamical stability perspective. In Proceedings of the 32Nd International Conference on Neural Information Processing Systems, NIPS'18, USA, 2018. Curran Associates Inc.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. CoRR, abs/1708.07747, 2017.  
Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. A Walk with SGD. arXiv e-prints, art. arXiv:1802.08770, Feb 2018.  
Greg Yang, Jeffrey Pennington, Vinay Rao, Jascha Sohl-Dickstein, and Samuel S. Schoenholz. A mean field theory of batch normalization. CoRR, abs/1902.08129, 2019.  
Zhewei Yao, Amir Gholami, Qi Lei, Kurt Keutzer, and Michael W. Mahoney. Hessian-based analysis of large batch training and robustness to adversaries. CoRR, abs/1802.08241, 2018.  
Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma. The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior of Escaping from Minima and Regularization Effects. arXiv eprints, art. arXiv:1803.00195, Feb 2018.
