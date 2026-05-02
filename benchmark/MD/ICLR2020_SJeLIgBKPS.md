# GRADIENT DESCENT MAXIMIZES THE MARGIN OF HOMOGENEOUS NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

In this paper, we study the implicit regularization of the gradient descent algorithm in homogeneous neural networks, including fully-connected and convolutional neural networks with ReLU or LeakyReLU activations. In particular, we study the gradient descent or gradient flow (i.e., gradient descent with infinitesimal step size) optimizing the logistic loss or cross-entropy loss of any homogeneous model (possibly non-smooth), and show that if the training loss decreases below a certain threshold, then we can define a smoothed version of the normalized margin which increases over time. We also formulate a natural constrained optimization problem related to margin maximization, and prove that both the normalized margin and its smoothed version converge to the objective value at a KKT point of the optimization problem. Our results generalize the previous results for logistic regression with one-layer or multi-layer linear networks, and provide more quantitative convergence results with weaker assumptions than previous results for homogeneous smooth neural networks. We conduct several experiments to justify our theoretical finding on MNIST and CIFAR-10 datasets. Finally, as margin is closely related to robustness, we discuss potential benefits of training longer for improving the robustness of the model.

# 1 INTRODUCTION

A major open question in deep learning is why gradient descent or its variants, are biased towards solutions with good generalization performance on the test set. To achieve a better understanding, previous works have studied the implicit bias of gradient descent in different settings. One simple but insightful setting is linear logistic regression on linearly separable data. In this setting, the model is parameterized by a weight vector  $\boldsymbol{w}$ , and the class prediction for any data point  $\boldsymbol{x}$  is determined by the sign of  $\boldsymbol{w}^{\top} \boldsymbol{x}$ . Therefore, only the direction  $\boldsymbol{w} / \| \boldsymbol{w} \|_2$  is important for making prediction. Soudry et al. (2018a,b); Ji and Telgarsky (2018); Nacson et al. (2018) investigated this problem and proved that the direction of  $\boldsymbol{w}$  converges to the direction that maximizes the  $L^2$ -margin while the norm of  $\boldsymbol{w}$  diverges to  $+\infty$ , if we train  $\boldsymbol{w}$  with (stochastic) gradient descent on logistic loss. Interestingly, this convergent direction is the same as that of any regularization path: any sequence of weight vectors  $\{\boldsymbol{w}_t\}$  such that every  $\boldsymbol{w}_t$  is a global minimum of the  $L^2$ -regularized loss  $\mathcal{L}(\boldsymbol{w}) + \frac{\lambda_t}{2} \| \boldsymbol{w} \|_2^2$  with  $\lambda_t \to 0$  (Rosset et al., 2004). Indeed, the trajectory of gradient descent is also pointwise close to a regularization path (Suggala et al., 2018).

The aforementioned linear logistic regression can be viewed as a single-layer neural network. A natural and important question is to what extent gradient descent has similar implicit bias for modern deep neural networks. For theoretical analysis, a natural candidate is to consider homogeneous neural networks. Here a neural network  $\Phi$  is said to be (positively) homogeneous if there is a number  $L > 0$  (called the order) such that the network output  $\Phi (\theta ;\boldsymbol {x})$ , where  $\theta$  stands for the parameter and  $\boldsymbol{x}$  stands for the input, satisfies the following:

$$
\forall c > 0: \Phi (c \boldsymbol {\theta}; \boldsymbol {x}) = c ^ {L} \Phi (\boldsymbol {\theta}; \boldsymbol {x}) \text {f o r a l l} \boldsymbol {\theta} \text {a n d} \boldsymbol {x}. \tag {1}
$$

It is important to note that many neural networks are homogeneous (Neyshabur et al., 2015; Du et al., 2018). For example, deep fully-connected neural networks or deep CNNs with ReLU or LeakyReLU activations can be made homogeneous if we remove all the bias terms, and the order  $L$  is exactly equal to the number of layers.

In (Wei et al., 2018), it is shown that the regularization path does converge to the max-margin direction for homogeneous neural networks with cross-entropy or logistic loss. This result suggests that gradient descent or gradient flow may also converge to the max-margin direction by assuming homogeneity, and this is indeed true for some sub-classes of homogeneous neural networks. For gradient flow, this convergent direction is proven for linear fully-connected networks (Ji and Telgarsky, 2019a). For gradient descent on linear fully-connected and convolutional networks, (Gunasekar et al., 2018b) formulate a constrained optimization problem related to margin maximization and prove that gradient descent converges to the direction of a KKT point or even the max-margin direction, under various assumptions including the convergence of loss and gradient directions. In an independent work, (Nacson et al., 2019a) generalize the result in (Gunasekar et al., 2018b) to smooth homogeneous models (we will discuss this work in more details in Section 2).

# 1.1 MAIN RESULTS

In this paper, we identify a minimal set of assumptions for proving our theoretical results for homogeneous neural networks on classification tasks. Besides homogeneity, we make two additional assumptions:

1. Exponential-type Loss Function. We require the loss function to have certain exponential tail (see Appendix A for the details). This assumption is not restrictive as it includes the most popular classification losses: exponential loss, logistic loss and cross-entropy loss.  
2. Separability. The neural network can separate the training data during training (i.e., the neural network can achieve  $100\%$  training accuracy)<sup>1</sup>.

While the first assumption is natural, the second requires some explanation. In fact, we assume that at some time  $t_0$ , the training loss is smaller than a threshold, and the threshold here is chosen to be so small that the training accuracy is guaranteed to be  $100\%$  (e.g., for the logistic loss and cross-entropy loss, the threshold can be set to  $\ln 2$ ). Empirically, state-of-the-art CNNs for image classification can even fit randomly labeled data easily (Zhang et al., 2017). Recent theoretical work on over-parameterized neural networks (Allen-Zhu et al., 2019; Zou et al., 2018) show that gradient descent can fit the training data if the width is large enough. Furthermore, in order to study the margin, ensuring the training data can be separated is inevitable; otherwise, there is no positive margin between the data and decision boundary.

Our Contribution. Similar to linear models, for homogeneous models, only the direction of parameter  $\pmb{\theta}$  is important for making predictions, and one can see that the margin  $\gamma (\pmb {\theta})$  scales linearly with  $\| \pmb {\theta}\| _2^L$ , when fixing the direction of  $\pmb{\theta}$ . To compare margins among  $\pmb{\theta}$  in different directions, it makes sense to study the normalized margin,  $\bar{\gamma} (\pmb {\theta}):= \gamma (\pmb {\theta}) / \| \pmb {\theta}\| _2^L$ .

In this paper, we focus on the training dynamics of the network after  $t_0$  (recall that  $t_0$  is a time that the training loss is less than the threshold). Our theoretical results can answer the following questions regarding the normalized margin.

First, how does the normalized margin change during training? The answer may seem complicated since one can easily come up with examples in which  $\bar{\gamma}$  increases or decreases in a short time interval. However, we can show that the overall trend of the normalized margin is to increase in the following sense: there exists a smoothed version of the normalized margin, denoted as  $\tilde{\gamma}$ , such that (1)  $|\tilde{\gamma} - \bar{\gamma}| \to 0$  as  $t \to \infty$ ; and (2)  $\tilde{\gamma}$  is non-decreasing for  $t > t_0$ .

Second, how large is the normalized margin at convergence? To answer this question, we formulate a natural constrained optimization problem which aims to directly maximize the margin. We show that every limit point of  $\{\pmb {\theta}(t) / \| \pmb {\theta}(t)\| _2:t > 0\}$  is along the direction of a KKT point of the max-margin problem. This indicates that gradient descent/gradient flow performs margin maximization implicitly in deep homogeneous networks. This result can be seen as a significant generalization of previous works (Soudry et al., 2018a;b; Ji and Telgarsky, 2019a; Gunasekar et al., 2018b) from linear classifiers to homogeneous classifiers.

As by-products of the above results, we derive tight asymptotic convergence/growth rates of the loss and weights. It is shown in (Soudry et al., 2018a;b; Ji and Telgarsky, 2018) that the loss decreases

![](images/ab6c44bef82b85de95d37e49f7f17f03b4991a74ff755400f411c31027bed420.jpg)  
(a)

![](images/7fe2bc08ec0cd2a1408c5fc1782ed31eccd73cf327f1b7a620179a9cbc0d9f17.jpg)  
(b)  
Figure 1: (a) Training CNNs with and without bias on MNIST, using SGD with learning rate 0.01. The training loss (left) decreases over time, and the normalized margin (right) keeps increasing after the model is fitted, but the growth rate is slow ( $\approx 1.8 \times 10^{-4}$  after 10000 epochs). (b) Training CNNs with and without bias on MNIST, using SGD with the loss-based learning rate scheduler. The training loss (left) decreases exponentially over time ( $< 10^{-800}$  after 9000 epochs), and the normalized margin (right) increases rapidly after the model is fitted ( $\approx 1.2 \times 10^{-3}$  after 10000 epochs,  $10 \times$  larger than that of SGD with learning rate 0.01). Experimental details are in Appendix J.

at the rate of  $O(1 / t)$ , the weight norm grows as  $O(\log t)$  for linear logistic regression. In this work, we generalize the result by showing that the loss decreases at the rate of  $O(1 / (t(\log t)^{2 - 2 / L}))$  and the weight norm grows as  $O((\log t)^{1 / L})$  for homogeneous neural networks with exponential loss, logistic loss, or cross-entropy loss.

Experiments. The main practical implication of our theoretical result is that training longer can enlarge the normalized margin. To justify this claim empirically, we train CNNs on MNIST and CIFAR-10 with SGD (see Section J.1). Results on MNIST are presented in Figure 1. For constant step size, we can see that the normalized margin keeps increasing, but the growth rate is rather slow (because the gradient gets smaller and smaller). Inspired by our convergence results for gradient descent, we use a learning rate scheduling method which enlarges the learning rate according to the current training loss, then the training loss decreases exponentially faster and the normalized margin increases significantly faster as well.

For feedforward neural networks with ReLU activation, the normalized margin on a training sample is closely related to the  $L^2$ -robustness (the  $L^2$ -distance from the training sample to the decision boundary). Indeed, the former divided by a Lipschitz constant is a lower bound for the latter. For example, the normalized margin is a lower bound for the  $L^2$ -robustness on fully-connected networks with ReLU activation (see, e.g., Theorem 4 in (Sokolic et al., 2017)). This fact suggests that training longer may have potential benefits on improving the robustness of the model. In our experiments, we observe noticeable improvements of  $L^2$ -robustness on both training and test sets (see Section J.2).

# 2 RELATED WORK

Implicit Bias in Training Linear Classifiers. For linear logistic regression on linearly separable data, Soudry et al. (2018a;b) showed that full-batch gradient descent converges in the direction of the max  $L^2$ -margin solution of the corresponding hard-margin Support Vector Machine (SVM). Subsequent works extended this result in several ways: Nacson et al. (2018) extended the results to the case of stochastic gradient descent; Gunasekar et al. (2018a) considered other optimization methods; Nacson et al. (2019b) considered other loss functions including those with poly-exponential tails; Ji and Telgarsky (2018) characterized the convergence of weight direction without assuming separability; Ji and Telgarsky (2019b) proved a tighter convergence rate for the weight direction.

Those results on linear logistic regression have been generalized to deep linear networks. Ji and Telgarsky (2019a) showed that the product of weights in a deep linear network with strictly decreasing loss converges in the direction of the max  $L^2$ -margin solution. Gunasekar et al. (2018b) showed more general results for gradient descent on linear fully-connected and convolutional networks with exponential loss, under various assumptions on the convergence of the loss and gradient direction.

Margin maximization phenomenon is also studied for boosting methods (Schapire et al., 1998; Schapire and Freund, 2012; Shalev-Shwartz and Singer, 2010; Telgarsky, 2013) and Normalized Perceptron (Ramdas and Pena, 2016).

Implicit Bias in Training Nonlinear Classifiers. Soudry et al. (2018a) analyzed the case where there is only one trainable layer of a ReLU network. Xu et al. (2018) characterized the implicit bias for the model consisting of one single ReLU unit. Our work is closely related to a recent independent work by (Nacson et al., 2019a) which we discuss in details below.

Comparison with (Nacson et al., 2019a). Very Recently, (Nacson et al., 2019a) analyzed gradient descent for smooth homogeneous models and proved the convergence of parameter direction to a KKT point of the aforementioned max-margin problem. Compared with their work, our work adopts much weaker assumptions: (1) They assume the training loss converges to 0, but in our work we only require that the training loss is lower than a small threshold value at some time  $t_0$  (and we prove the exact convergence rate of the loss after  $t_0$ ); (2) They assume the convergence of parameter direction $^2$ , while we prove that KKT conditions hold for all limit points of  $\{\pmb{\theta}(t) / \| \pmb{\theta}(t) \|_2 : t > 0\}$ , without requiring any convergence assumption; (3) They assume the convergence of the direction of losses (the direction of the vector whose entries are loss values on every data point) and Linear Independence Constraint Qualification (LICQ) for the max-margin problem, while we do not need such assumptions. Besides the above differences in assumptions, we also prove the monotonicity of the normalized margin and provide tight convergence rate for training loss. We believe both results are interesting in their own right.

Another technical difference is that their work analyzes discrete gradient descent on smooth homogeneous models (which fails to capture ReLU networks). In our work, we analyze both gradient descent on smooth homogeneous models and also gradient flow on homogeneous models which could be non-smooth.

Other Works on Implicit Bias. Banburski et al. (2019) also studied the dynamics of gradient flow and among other things, provided mathematical insights to the implicit bias towards max margin solution for homogeneous networks. We note that their analysis of gradient flow decomposes the dynamics to the tangent component and radial component, which is similar to our proof of Theorem 4.1 in spirit. Wilson et al. (2017); Ali et al. (2018); Gunasekar et al. (2018a) showed that for the linear least-square problem gradient-based methods converge to the unique global minimum that is closest to the initialization in  $L^2$  distance. Du et al. (2019); Jacot et al. (2018); Lee et al. (2019); Arora et al. (2019b) showed that over-parameterized neural networks of sufficient width (or infinite width) behave as linear models with Neural Tangent Kernel (NTK) with proper initialization and gradient descent converges linearly to a global minimum near the initial point. Other related works include (Ma et al., 2017; Gidel et al., 2019; Arora et al., 2019a; Suggala et al., 2018; Blanc et al., 2019; Neyshabur et al., 2014; 2015).

# 3 PRELIMINARIES

Basic Notations. For any  $N \in \mathbb{N}$ , let  $[N] = \{1, \dots, N\}$ .  $\| \pmb{v} \|_2$  denotes the  $L^2$ -norm of a vector  $\pmb{v}$ . The default base of  $\log$  is  $e$ . For a function  $f: \mathbb{R}^d \to \mathbb{R}$ ,  $\nabla f(\pmb{x})$  stands for the gradient at  $\pmb{x}$  if it exists. A function  $f: X \to \mathbb{R}^d$  is  $\mathcal{C}^k$ -smooth if  $f$  is  $k$  times continuously differentiable. A function  $f: X \to \mathbb{R}$  is locally Lipschitz if for every  $\pmb{x} \in X$  there exists a neighborhood  $U$  of  $\pmb{x}$  such that the restriction of  $f$  on  $U$  is Lipschitz continuous.

Non-smooth Analysis. For a locally Lipschitz function  $f: X \to \mathbb{R}$ , the Clarke's subdifferential (Clarke, 1975; Clarke et al., 2008; Davis et al., 2019) at  $x \in X$  is the convex set  $\partial^{\circ}f(x) := \operatorname{conv}\left\{\lim_{k \to \infty} \nabla f(\boldsymbol{x}_k): \boldsymbol{x}_k \to \boldsymbol{x}, f \text{ is differentiable at } \boldsymbol{x}_k\right\}$ . For brevity, we say that a function  $z: I \to \mathbb{R}^d$  on the interval  $I$  is an arc if  $z$  is absolutely continuous for any compact sub-interval of  $I$ . For an arc  $z, z'(t)$  (or  $\frac{dz}{dt}(t)$ ) stands for the derivative at  $t$  if it exists. Following the terminology in (Davis et al., 2019), we say that a locally Lipschitz function  $f: \mathbb{R}^d \to \mathbb{R}$  admits a chain rule if

for any arc  $z:[0, + \infty)\to \mathbb{R}^d$ $\forall \pmb {h}\in \partial^{\circ}f(z(t)):(f\circ \pmb {z})^{\prime}(t) = \langle \pmb {h},\pmb{z}^{\prime}(t)\rangle$  holds for a.e.  $t > 0$  (see also Appendix H).

Binary Classification. Let  $\Phi$  be a neural network, assumed to be parameterized by  $\pmb{\theta}$ . The output of  $\Phi$  on an input  $\pmb{x} \in \mathbb{R}^{d_{\mathrm{x}}}$  is a real number  $\Phi(\pmb{\theta}; \pmb{x})$ , and the sign of  $\Phi(\pmb{\theta}; \pmb{x})$  stands for the classification result. A dataset is denoted by  $\mathcal{D} = \{(x_n, y_n) : n \in [N]\}$ , where  $x_n \in \mathbb{R}^{d_{\mathrm{x}}}$  stands for a data input and  $y_n \in \{\pm 1\}$  stands for the corresponding label. For a loss function  $\ell: \mathbb{R} \to \mathbb{R}$ , we define the training loss of  $\Phi$  on the dataset  $\mathcal{D}$  to be  $\mathcal{L}(\pmb{\theta}) := \sum_{n=1}^{N} \ell(y_n \Phi(\pmb{\theta}; x_n))$ .

Gradient Descent. We consider the process of training this neural network  $\Phi$  with either gradient descent or gradient flow. For gradient descent, we assume the training loss  $\mathcal{L}(\pmb{\theta})$  is  $\mathcal{C}^2$ -smooth and describe the gradient descnet process as  $\pmb{\theta}(t+1) = \pmb{\theta}(t) - \eta(t)\nabla \mathcal{L}(\pmb{\theta}(t))$ , where  $\eta(t)$  is the learning rate at time  $t$  and  $\nabla \mathcal{L}(\pmb{\theta}(t))$  is the gradient of  $\mathcal{L}$  at  $\pmb{\theta}(t)$ .

Gradient Flow. For gradient flow, we do not assume the differentiability but only some regularity assumptions including locally Lipschitz. Gradient flow can be seen as gradient descent with infinitesimal step size. In this model,  $\pmb{\theta}$  changes continuously with time, and the trajectory of parameter  $\pmb{\theta}$  during training is an arc  $\pmb{\theta}:[0, +\infty)\to \mathbb{R}^d,t\mapsto \pmb{\theta}(t)$  that satisfies the differential inclusion  $\frac{d\pmb{\theta}(t)}{dt}\in -\partial^\circ \mathcal{L}(\pmb {\theta}(t))$  for a.e.  $t\geq 0$ . The Clarke's subdifferential  $\partial^{\circ}\mathcal{L}$  is a natural generalization of the usual differential to non-differentiable functions. If  $\mathcal{L}(\pmb {\theta})$  is actually a  $\mathcal{C}^1$ -smooth function, the above differential inclusion reduces to  $\frac{d\pmb{\theta}(t)}{dt} = -\nabla \mathcal{L}(\pmb {\theta}(t))$  for all  $t\geq 0$ , which corresponds to the gradient flow with differential in the usual sense.

# 4 GRADIENT DESCENT / GRADIENT FLOW ON HOMOGENEOUS MODEL

In this section, we first state our results for gradient flow and gradient descent on homogeneous models with exponential loss  $\ell(q) \coloneqq e^{-q}$  for simplicity of presentation. Due to space limit, we defer the more general results which hold for a large family of loss functions (including logistic loss and cross-entropy loss) to Appendix A and F.

# 4.1 ASSUMPTIONS

Gradient Flow. For gradient flow, we assume the following:

(A1). (Regularity). For any fixed  $x$ ,  $\Phi(\cdot; x)$  is locally Lipschitz and admits a chain rule;  
(A2). (Homogeneity). There exists  $L > 0$  such that  $\forall \alpha > 0 : \Phi(\alpha \theta; x) = \alpha^L \Phi(\theta; x)$ ;  
(A3). (Exponential Loss).  $\ell(q) = e^{-q}$ ;  
(A4). (Separability). There exists a time  $t_0$  such that  $\mathcal{L}(\pmb{\theta}(t_0)) < 1$ .

(A1) is a technical assumption about the regularity of the network output. As shown in (Davis et al., 2019), the output of almost every neural network admits a chain rule (as long as the neural network is composed by definable pieces in an o-minimal structure, e.g., ReLU, sigmoid, LeakyReLU).

(A2) assumes the homogeneity, the main property we assume in this work. (A3), (A4) correspond to the two conditions introduced in Section 1. The exponential loss in (A3) is a main focus of this section, and more general results are in Appendix A and F. (A4) is a separability assumption: the condition  $\mathcal{L}(\pmb{\theta}(t_0)) < 1$  ensures that  $\ell(y_n \Phi(\pmb{\theta}(t_0); \pmb{x}_n)) < 1$  for all  $n \in [N]$ , and thus  $y_n \Phi(\pmb{\theta}(t_0); \pmb{x}_n) > 0$ , meaning that  $\Phi$  classifies every  $\pmb{x}_n$  correctly.

Gradient Descent. For gradient descent, we assume (A2), (A3), (A4) similarly as for gradient flow, and the following two assumptions (S1) and (S5).

(S1). (Smoothness). For any fixed  $\pmb{x}$ ,  $\Phi(\cdot; \pmb{x})$  is  $\mathcal{C}^2$ -smooth on  $\mathbb{R}^d \setminus \{\mathbf{0}\}$ .  
(S5). (Learning rate condition, Informal).  $\eta(t) = \eta_0$  for a sufficiently small constant  $\eta_0$ . In fact,  $\eta(t)$  is even allowed to be as large as  $O(\mathcal{L}(t)^{-1}\mathrm{polylog}\frac{1}{\mathcal{L}(t)})$ . See Appendix E.1 for the details.

(S5) is natural since deep neural networks are usually trained with constant learning rates. (S1) ensures the smoothness of  $\Phi$ , which is often assumed in the optimization literature in order to analyze

gradient descent. While (S1) does not hold for neural networks with ReLU, it does hold for neural networks with smooth homogeneous activation such as the quadratic activation  $\phi(x) \coloneqq x^2$  (Li et al., 2018b; Du and Lee, 2018) or powers of ReLU  $\phi(x) \coloneqq \mathrm{ReLU}(x)^{\alpha}$  for  $\alpha > 2$  (Zhong et al., 2017; Klusowski and Barron, 2018; Li et al., 2019).

# 4.2 MAIN THEOREM: MONOTONICITY OF NORMALIZED MARGINS

The margin for a single data point  $(\pmb{x}_n, y_n)$  is defined to be  $q_n(\pmb{\theta}) \coloneqq y_n \Phi(\pmb{\theta}; \pmb{x}_n)$ , and the margin for the entire dataset is defined to be  $q_{\min}(\pmb{\theta}) \coloneqq \min_{n \in [N]} q_n(\pmb{\theta})$ . By homogeneity, the margin  $q_{\min}(\pmb{\theta})$  scales linearly with  $\|\pmb{\theta}\|_2^L$  for any fixed direction since  $q_{\min}(c\pmb{\theta}) = c^L q_{\min}(\pmb{\theta})$ . So we consider the normalized margin defined as below:

$$
\bar {\gamma} (\boldsymbol {\theta}) := q _ {\min } \left(\frac {\boldsymbol {\theta}}{\| \boldsymbol {\theta} \| _ {2}}\right) = \frac {q _ {\min } (\boldsymbol {\theta})}{\| \boldsymbol {\theta} \| _ {2} ^ {L}}. \tag {2}
$$

We say  $f$  is an  $\epsilon$ -additive approximation for the normalized margin if  $\bar{\gamma} - \epsilon \leq f \leq \bar{\gamma}$ , and  $c$ -multiplicative approximation if  $c\bar{\gamma} \leq f \leq \bar{\gamma}$ .

Gradient Flow. Our first result is on the overall trend of the normalized margin  $\bar{\gamma} (\theta (t))$ . For both gradient flow and gradient descent, we identify a smoothed version of the normalized margin, and show that it is non-decreasing during training. More specifically, we have the following theorem for gradient flow.

Theorem 4.1 (Corollary of Theorem A.7). Under assumptions (A1) - (A4), there exists an  $O(\|\pmb{\theta}\|_2^{-L})$ -additive approximation function  $\tilde{\gamma}(\pmb{\theta})$  for the normalized margin such that the following statements are true for gradient flow:

1. For a.e.  $t > t_0$ ,  $\frac{d}{dt}\tilde{\gamma} (\pmb {\theta}(t))\geq 0$  
2. For a.e.  $t > t_0$ , either  $\frac{d}{dt}\tilde{\gamma} (\pmb {\theta}(t)) > 0$  or  $\frac{d}{dt}\frac{\pmb{\theta}(t)}{\|\pmb{\theta}(t)\|_2} = 0$ ;  
3.  $\mathcal{L}(\pmb {\theta}(t))\to 0$  and  $\| \pmb {\theta}(t)\| _2\to \infty$  as  $t\rightarrow +\infty$  ; therefore,  $|\tilde{\gamma} (\pmb {\theta}(t)) - \tilde{\gamma} (\pmb {\theta}(t))|\to 0.$

More concretely, the function  $\tilde{\gamma} (\pmb {\theta})$  in Theorem 4.1 is defined as

$$
\tilde {\gamma} (\boldsymbol {\theta}) := \frac {\log \frac {1}{\mathcal {L} (\boldsymbol {\theta})}}{\| \boldsymbol {\theta} \| _ {2} ^ {L}} = \frac {- \log \left(\sum_ {n = 1} ^ {N} e ^ {- q _ {n} (\boldsymbol {\theta})}\right)}{\| \boldsymbol {\theta} \| _ {2} ^ {L}}. \tag {3}
$$

Note that the only difference between  $\bar{\gamma}(\pmb{\theta})$  and  $\tilde{\gamma}(\pmb{\theta})$  is that  $q_{\min}(\pmb{\theta})$  in  $\bar{\gamma}(\pmb{\theta})$  is replaced by  $\log \frac{1}{\mathcal{L}(\pmb{\theta})} = -\mathrm{LSE}(-q_1(\pmb{\theta}), \dots, -q_N(\pmb{\theta}))$ , where  $\mathrm{LSE}(a_1, \dots, a_N) = \log (\exp(a_1) + \dots + \exp(a_N))$  is the LogSumExp function. This is indeed a very natural idea, and previous works on linear models (e.g., (Telgarsky, 2013; Nacson et al., 2019b)) also approximate  $q_{\min}$  with LogSumExp in the analysis of margin. It is easy to see why  $\tilde{\gamma}(\pmb{\theta})$  is an  $O(\|\pmb{\theta}\|_2^{-L})$ -additive approximation for  $\bar{\gamma}(\pmb{\theta})$ :  $e^{a_{\max}} \leq \sum_{n=1}^{N} e^{a_n} \leq Ne^{a_{\max}}$  holds for  $a_{\max} = \max \{a_1, \dots, a_N\}$ , so  $a_{\max} \leq \mathrm{LSE}(a_1, \dots, a_N) \leq a_{\max} + \log N$ ; combining this with the definition of  $\tilde{\gamma}(\pmb{\theta})$  gives  $\bar{\gamma}(\pmb{\theta}) - \|\pmb{\theta}\|_2^{-L} \log N \leq \tilde{\gamma}(\pmb{\theta}) \leq \bar{\gamma}(\pmb{\theta})$ .

Gradient Descent. For gradient descent, Theorem 4.1 holds similarly with a slightly different function  $\hat{\gamma} (\pmb {\theta})$  that approximates  $\bar{\gamma} (\pmb {\theta})$  multiplicatively rather than additively.

Theorem 4.2 (Corollary of Theorem E.2). Under assumptions (S1), (A2) - (A4), (S5), there exists an  $(1 - O(1 / (\log \frac{1}{\mathcal{L}})))$ -multiplicative approximation function  $\hat{\gamma}(\theta)$  for the normalized margin such that the following statements are true for gradient descent:

1. For all  $t > t_0$ ,  $\hat{\gamma}(\pmb{\theta}(t + 1)) \geq \hat{\gamma}(\pmb{\theta}(t))$ ;  
2. For all  $t > t_0$ , either  $\hat{\gamma}(\pmb{\theta}(t + 1)) > \hat{\gamma}(\pmb{\theta}(t))$  or  $\frac{\pmb{\theta}(t + 1)}{\|\pmb{\theta}(t + 1)\|_2} = \frac{\pmb{\theta}(t)}{\|\pmb{\theta}(t)\|_2}$ ;  
3.  $\mathcal{L}(\pmb {\theta}(t))\to 0$  and  $\| \pmb {\theta}(t)\| _2\to \infty$  as  $t\to +\infty$  ; therefore,  $|\widetilde{\gamma} (\pmb {\theta}(t)) - \hat{\gamma} (\pmb {\theta}(t))|\to 0.$

Due to the discreteness of gradient descent, the explicit formula for  $\hat{\gamma} (\pmb {\theta})$  is somewhat technical, and we refer the readers to Appendix E for full details.

Convergence Rates. It is shown in Theorem 4.1, 4.2 that  $\mathcal{L}(\pmb{\theta}(t)) \to 0$  and  $\| \pmb{\theta}(t) \|_2 \to \infty$ . In fact, with a more refined analysis, we can prove tight loss convergence and weight growth rates using the monotonicity of normalized margins.

Theorem 4.3 (Corollary of Theorem A.10 and E.5). For gradient flow under assumptions (A1) - (A4) or gradient descent under assumptions (S1), (A2) - (A4), (S5), we have the following tight bounds for training loss and weight norm:

$$
\mathcal {L} (\boldsymbol {\theta} (t)) = \Theta \left(\frac {1}{T (\log T) ^ {2 - 2 / L}}\right) \quad a n d \quad \| \boldsymbol {\theta} (t) \| _ {2} = \Theta \left((\log T) ^ {1 / L}\right),
$$

where  $T = t$  for gradient flow and  $T = \sum_{\tau = t_0}^{t - 1}\eta (\tau)$  for gradient descent.

# 4.3 MAIN THEOREM: CONVERGENCE TO KKT POINTS

For gradient flow,  $\tilde{\gamma}$  is upper-bounded by  $\tilde{\gamma} \leq \bar{\gamma} \leq \sup \{q_n(\pmb{\theta}) : \| \pmb{\theta} \|_2 = 1\}$ . Combining this with Theorem 4.1 and the monotone convergence theorem, it is not hard to see that  $\lim_{t \to +\infty} \bar{\gamma}(\pmb{\theta}(t))$  and  $\lim_{t \to +\infty} \tilde{\gamma}(\pmb{\theta}(t))$  exist and equal to the same value. Using a similar argument, we can draw the same conclusion for gradient descent.

To understand the implicit regularization effect, a natural question arises: what optimality property does the limit of normalized margin have? To this end, we identify a natural constrained optimization problem related to margin maximization, and prove that  $\theta(t)$  directionally converges to its KKT points, as shown below. We note that we can extend this result to the finite time case, and show that gradient flow or gradient descent passes through an approximate KKT point after a certain amount of time. See Theorem A.9 in Appendix A and Theorem E.4 in Appendix E for the details. We will briefly review the definition of KKT points and approximate KKT points for a constraint optimization problem in Appendix C.1.

Theorem 4.4 (Corollary of Theorem A.8 and E.3). For gradient flow under assumptions (A1) - (A4) or gradient descent under assumptions (S1), (A2) - (A4), (S5), any limit point  $\bar{\theta}$  of  $\left\{\frac{\theta(t)}{||\theta(t)||_2}:t\geq 0\right\}$  is along the direction of a KKT point of the following constrained optimization problem  $(P)$ :

$$
\min  \quad \frac {1}{2} \| \boldsymbol {\theta} \| _ {2} ^ {2} \qquad \text {s . t .} \quad q _ {n} (\boldsymbol {\theta}) \geq 1 \qquad \forall n \in [ N ]
$$

That is, for any limit point  $\bar{\theta}$ , there exists a scaling factor  $\alpha > 0$  such that  $\alpha \bar{\theta}$  satisfies Karush-Kuhn-Tucker (KKT) conditions of  $(P)$ .

Minimizing (P) over its feasible region is equivalent to maximizing the normalized margin over all possible directions. The proof is as follows. Note that we only need to consider all feasible points  $\pmb{\theta}$  with  $q_{\mathrm{min}}(\pmb {\theta}) > 0$ . For a fixed  $\pmb{\theta}$ ,  $\alpha \pmb{\theta}$  is a feasible point of (P) iff  $\alpha \geq q_{\mathrm{min}}(\pmb {\theta})^{-1 / L}$ . Thus, the minimum objective value over all feasible points of (P) in the direction of  $\pmb{\theta}$  is  $\frac{1}{2}\| \pmb {\theta} / q_{\mathrm{min}}(\pmb {\theta})^{1 / L}\| _2^2 = \frac{1}{2}\bar{\gamma} (\pmb {\theta})^{-2 / L}$ . Taking minimum over all possible directions, we can conclude that if the maximum normalized margin is  $\bar{\gamma}_{*}$ , then the minimum objective of (P) is  $\frac{1}{2}\bar{\gamma}_{*}^{-2 / L}$ .

It can be proved that (P) satisfies the Mangasarian-Fromovitz Constraint Qualification (MFCQ) (See Lemma C.7). Thus, KKT conditions are first-order necessary conditions for global optimality. For linear models, KKT conditions are also sufficient for ensuring global optimality; however, for deep homogenous networks,  $q_{n}(\pmb{\theta})$  can be highly non-convex. Indeed, as gradient descent is a first-order optimization method, if we do not make further assumptions on  $q_{n}(\pmb{\theta})$ , then it is easy to construct examples that gradient descent does not lead to a normalized margin that is globally optimal. Thus, proving the convergence to KKT points is perhaps the best we can hope for in our setting, and it is an interesting future work to prove stronger convergence results with further natural assumptions.

Moreover, we can prove the following corollary, which characterizes the optimality of the normalized margin using SVM with Neural Tangent Kernel (NTK, introduced in (Jacot et al., 2018)) defined at limit points. The proof is deferred to Appendix C.6.

Corollary 4.5 (Corollary of Theorem 4.4). Assume (S1). Then for gradient flow under assumptions (A2) - (A4) or gradient descent under assumptions (A2) - (A4), (S5), any limit point  $\bar{\theta}$  of  $\{\pmb {\theta}(t) / \| \pmb {\theta}(t)\| _2:t\geq 0\}$  is along the max-margin direction for the hard-margin SVM with kernel

$K_{\bar{\theta}}(\pmb{x},\pmb{x}^{\prime}) = \left\langle \nabla \Phi_{\pmb{x}}(\bar{\theta}),\nabla \Phi_{\pmb{x}^{\prime}}(\bar{\theta})\right\rangle$  where  $\Phi_{\pmb{x}}(\pmb {\theta}):= \Phi (\pmb {\theta};\pmb {x})$  . That is, for some  $\alpha >0$ $\alpha \bar{\theta}$  is the optimal solution for the following constrained optimization problem:

$$
\min  \quad \frac {1}{2} \| \boldsymbol {\theta} \| _ {2} ^ {2} \quad s. t. \quad y _ {n} \left\langle \boldsymbol {\theta}, \nabla \Phi_ {\boldsymbol {x} _ {n}} (\bar {\boldsymbol {\theta}}) \right\rangle \geq 1 \quad \forall n \in [ N ]
$$

If we assume (A1) instead of (S1) for gradient flow, then there exists a mapping  $\pmb{h}(\pmb{x}) \in \partial^{\circ} \Phi_{\pmb{x}}(\bar{\pmb{\theta}})$  such that the same conclusion holds for  $K_{\bar{\pmb{\theta}}}(\pmb{x}, \pmb{x}') = \langle \pmb{h}(\pmb{x}), \pmb{h}(\pmb{x}') \rangle$ .

# 4.4 OTHER MAIN RESULTS

The above results can be extended to other settings. Here we discuss them in the context of gradient flow for simplicity, but it is not hard to generalize them to gradient descent.

Other Binary Classification Loss. The results on exponential loss can be generalized to a much broader class of binary classification loss. The class includes the logistic loss which is one of the most popular loss functions,  $\ell(q) = \log(1 + e^{-q})$ . The function class also includes other losses with exponential tail, e.g.,  $\ell(q) = e^{-q^3}$ ,  $\ell(q) = \log(1 + e^{-q^3})$ . For all those loss functions, we can use its inverse function  $\ell^{-1}$  to define the smoothed normalized margin as follows

$$
\tilde {\gamma} (\boldsymbol {\theta}) = \frac {\ell^ {- 1} (\mathcal {L} (\boldsymbol {\theta}))}{\| \boldsymbol {\theta} \| _ {2} ^ {L}}.
$$

Theorem 4.1 and 4.4 continue to hold for gradient flow. See Appendix A for the details.

Cross-entropy Loss. In multi-class classification, we can define  $q_{n}$  to be the difference between the classification score for the true label and the maximum score for the other labels, then the margin  $q_{\mathrm{min}} \coloneqq \min_{n \in [N]} q_{n}$  and the normalized margin  $\bar{\gamma}(\theta) \coloneqq \frac{q_{\mathrm{min}}(\theta)}{\|\theta\|_2^L}$  can be similarly defined as before. In Appendix F, we define the smoothed normalized margin for cross-entropy loss to be the same as that for logistic loss (See Remark A.4), and we show that Theorem 4.1 and Theorem 4.4 still hold (but with a slightly different definition of (P)) for gradient flow.

Multi-homogeneous Models. Some neural networks indeed possess a stronger property than homogeneity, which we call multi-homogeneity. For example, the output of a CNN (without bias terms) is 1-homogeneous with respect to the weights of each layer. In general, we say that a neural network  $\Phi(\pmb{\theta};\pmb{x})$  with  $\pmb{\theta} = (\pmb{w}_1,\dots,\pmb{w}_m)$  is  $(k_1,\ldots,k_m)$ -homogeneous if for any  $\pmb{x}$  and any  $c_1,\ldots,c_m > 0$ , we have  $\Phi(c_1\pmb{w}_1,\ldots,c_m\pmb{w}_m;\pmb{x}) = \prod_{i=1}^{m} c_i^{k_i} \cdot \Phi(\pmb{w}_1,\ldots,\pmb{w}_m;\pmb{x})$ . In the previous example, an  $L$ -layer CNN with layer weights  $\pmb{\theta} = (\pmb{w}_1,\dots,\pmb{w}_L)$  is  $(1,\dots,1)$ -homogeneous.

One can easily see that that  $(k_{1},\ldots ,k_{m})$  -homogeneity implies  $L$  -homogeneity, where  $L = \sum_{i = 1}^{m}k_{i}$ , so our previous analysis for homogeneous models still applies to multi-homogeneous models. But it would be better to define the normalized margin for multi-homogeneous model as

$$
\bar {\gamma} \left(\boldsymbol {w} _ {1}, \dots , \boldsymbol {w} _ {m}\right) := q _ {\min } \left(\frac {\boldsymbol {w} _ {1}}{\| \boldsymbol {w} _ {1} \| _ {2}}, \dots , \frac {\boldsymbol {w} _ {m}}{\| \boldsymbol {w} _ {m} \| _ {2}}\right) = \frac {q _ {\min }}{\prod_ {i = 1} ^ {m} \| \boldsymbol {w} _ {i} \| _ {2} ^ {k _ {i}}}. \tag {4}
$$

In this case, the smoothed approximation of  $\bar{\gamma}$  for general binary classification loss (under some conditions) can be similarly defined:

$$
\tilde {\gamma} \left(\boldsymbol {w} _ {1}, \dots , \boldsymbol {w} _ {m}\right) := \frac {\ell^ {- 1} (\mathcal {L})}{\prod_ {i = 1} ^ {m} \| \boldsymbol {w} _ {i} \| _ {2} ^ {k _ {i}}}, \tag {5}
$$

It can be shown that  $\tilde{\gamma}$  is also non-decreasing during training when the loss is small enough (Appendix G). In the case of cross-entropy loss, we can still define  $\tilde{\gamma}$  by (5) while  $\ell(\cdot)$  is set to the logistic loss in the formula.

# 5 PROOF SKETCH: GRADIENT FLOW ON HOMOGENEOUS MODEL WITH EXPONENTIAL LOSS

In this section, we present a proof sketch in the case of gradient flow on homogeneous model with exponential loss to illustrate our proof ideas. Due to space limit, the proof for the main theorems on gradient flow and gradient descent in Section 4 are deferred to Appendix A and E respectively.

For convenience, we introduce a few more notations for a  $L$ -homogeneous neural network  $\Phi(\pmb{\theta}; \pmb{x})$ . Let  $\mathcal{S}^{d-1} = \{\pmb{\theta} \in \mathbb{R}^d : \| \pmb{\theta} \|_2 = 1\}$  be the set of  $L^2$ -normalized parameters. Define  $\rho := \| \pmb{\theta} \|_2$  and  $\hat{\pmb{\theta}} := \frac{\pmb{\theta}}{\|\pmb{\theta}\|_2} \in \mathcal{S}^{d-1}$  to be the length and direction of  $\pmb{\theta}$ . For both gradient descent and gradient flow,  $\pmb{\theta}$  is a function of time  $t$ . For convenience, we also view the functions of  $\pmb{\theta}$ , including  $\mathcal{L}(\pmb{\theta}), q_n(\pmb{\theta}), q_{\min}(\pmb{\theta})$ , as functions of  $t$ . So we can write  $\mathcal{L}(t) := \mathcal{L}(\pmb{\theta}(t)), q_n(t) := q_n(\pmb{\theta}(t)), q_{\min}(t) := q_{\min}(\pmb{\theta}(t))$ .

Lemma 5.1 below is the key lemma in our proof. It decomposes the growth of the smoothed normalized margin into the ratio of two quantities related to the radial and tangential velocity components of  $\theta$  respectively. We will give a proof sketch for this later in this section. We believe that this lemma is of independent interest.

Lemma 5.1 (Corollary of Lemma B.1). For a.e.  $t > t_0$

$$
\frac {d}{d t} \log \rho > 0 a n d \frac {d}{d t} \log \tilde {\gamma} \geq L \left(\frac {d}{d t} \log \rho\right) ^ {- 1} \left\| \frac {d \hat {\boldsymbol {\theta}}}{d t} \right\| _ {2} ^ {2}.
$$

Using Lemma 5.1, the first two claims in Theorem 4.1 can be directly proved. For the third claim, we make use of the monotonicity of the margin to lower bound the gradient, and then show  $\mathcal{L} \to 0$  and  $\rho \to +\infty$ . Recall that  $\tilde{\gamma}$  is an  $O(\rho^{-L})$ -additive approximation for  $\tilde{\gamma}$ . So this proves the third claim. We defer the detailed proof to Appendix B.

To show Theorem 4.4, we first change the time measure to  $\log \rho$ , i.e., now we see  $t$  as a function of  $\log \rho$ . So the second inequality in Lemma 5.1 can be rewritten as  $\frac{d\log\tilde{\gamma}}{d\log\rho} \geq L\|\frac{d\hat{\theta}}{d\log\rho}\|_2^2$ . Integrating on both sides and noting that  $\tilde{\gamma}$  is upper-bounded, we know that there must be many instant  $\log \rho$  such that  $\|\frac{d\hat{\theta}}{d\log\rho}\|_2$  is small. By analyzing the landscape of training loss, we show that these points are "approximate" KKT points. Then we show that every convergent sub-sequence of  $\{\hat{\theta}(t): t \geq 0\}$  can be modified to be a sequence of "approximate" KKT points which converges to the same limit. Then we conclude the proof by applying a theorem from (Dutta et al., 2013) to show that the limit of this convergent sequence of "approximate" KKT points is a KKT point. We defer the detailed proof to Appendix C.

Now we give a proof sketch for Lemma 5.1, in which we derive the formula of  $\tilde{\gamma}$  step by step. In the proof, we obtain several clean close form formulas for several relevant quantities, by using the chain rule and Euler's theorem for homogenous functions extensively.

Proof Sketch of Lemma 5.1. For ease of presentation, we ignore the regularity issues of taking derivatives in this proof sketch. We start from the equation  $\frac{d\mathcal{L}}{dt} = -\left\langle \partial^{\circ}\mathcal{L}(\pmb {\theta}(t)),\frac{d\pmb{\theta}}{dt}\right\rangle = -\left\| \frac{d\pmb{\theta}}{dt}\right\|_{2}^{2}$  which follows from the chain rule (see also Lemma H.3). Then we note that  $\frac{d\pmb{\theta}}{dt}$  can be decomposed into two parts: the radial component  $\pmb {v}\coloneqq \hat{\pmb{\theta}}\hat{\pmb{\theta}}^{\top}\frac{d\pmb{\theta}}{dt}$  and the tangent component  $\pmb {u}\coloneqq (\pmb {I} - \hat{\pmb{\theta}}\hat{\pmb{\theta}}^{\top})\frac{d\pmb{\theta}}{dt}$ .

The radial component is easier to analyze. By the chain rule,  $\| \pmb{v}\| _2 = \hat{\pmb{\theta}}^\top \frac{d\pmb{\theta}}{dt} = \frac{1}{\rho}\left\langle \pmb {\theta},\frac{d\pmb{\theta}}{dt}\right\rangle = \frac{1}{\rho}\cdot \frac{1}{2}\frac{d\rho^2}{dt}$ . For  $\frac{1}{2}\frac{d\rho^2}{dt}$ , we have an exact formula:

$$
\frac {1}{2} \frac {d \rho^ {2}}{d t} = \left\langle \boldsymbol {\theta}, \frac {d \boldsymbol {\theta}}{d t} \right\rangle = \left\langle \sum_ {n = 1} ^ {N} e ^ {- q _ {n}} \partial^ {\circ} q _ {n}, \boldsymbol {\theta} \right\rangle = L \sum_ {n = 1} ^ {N} e ^ {- q _ {n}} q _ {n}, \tag {6}
$$

where the last equality is due to  $\langle \partial^\circ q_n,\pmb {\theta}\rangle = Lq_n$  by homogeneity of  $q_{n}$ . This equation is sometimes called Euler's theorem for homogeneous functions (see Theorem B.2). For differentiable functions, it can be easily proved by taking the derivative over  $c$  on both sides of  $q_{n}(c\pmb {\theta}) = c^{L}q_{n}(\pmb {\theta})$  and letting  $c = 1$ .

With (6), we can lower bound  $\frac{1}{2}\frac{d\rho^2}{dt}$  by

$$
\frac {1}{2} \frac {d \rho^ {2}}{d t} = L \sum_ {n = 1} ^ {N} e ^ {- q _ {n}} q _ {n} \geq L \sum_ {n = 1} ^ {N} e ^ {- q _ {n}} q _ {\min } \geq L \cdot \mathcal {L} \log \frac {1}{\mathcal {L}}, \tag {7}
$$

where the last inequality uses the fact that  $e^{-q_{\mathrm{min}}} \leq \mathcal{L}$ . (7) also implies that  $\frac{1}{2}\frac{d\rho^2}{dt} > 0$  for  $t > t_0$  since  $\mathcal{L}(t_0) < 1$  and  $\mathcal{L}$  is non-increasing. As  $\frac{d}{dt}\log \rho = \frac{1}{2\rho^2}\frac{d\rho^2}{dt}$ , this also proves the first inequality of Lemma 5.1.

Now, we have  $\| \pmb{v} \|_2^2 = \frac{1}{\rho^2} \left( \frac{1}{2} \frac{d\rho^2}{dt} \right)^2 = \frac{1}{2} \frac{d\rho^2}{dt} \cdot \frac{d}{dt} \log \rho$  on the one hand; on the other hand, by the chain rule we have  $\frac{d\hat{\pmb{\theta}}}{dt} = \frac{1}{\rho^2} (\rho \frac{d\pmb{\theta}}{dt} - \frac{d\rho}{dt} \pmb{\theta}) = \frac{1}{\rho^2} (\rho \frac{d\pmb{\theta}}{dt} - (\hat{\pmb{\theta}}^\top \frac{d\pmb{\theta}}{dt}) \pmb{\theta}) = \frac{\pmb{u}}{\rho}$ . So we have

$$
- \frac {d \mathcal {L}}{d t} = \left\| \frac {d \pmb {\theta}}{d t} \right\| _ {2} ^ {2} = \| \pmb {v} \| _ {2} ^ {2} + \| \pmb {u} \| _ {2} ^ {2} = \frac {1}{2} \frac {d \rho^ {2}}{d t} \cdot \frac {d}{d t} \log \rho + \rho^ {2} \left\| \frac {d \hat {\pmb {\theta}}}{d t} \right\| _ {2} ^ {2}
$$

Dividing  $\frac{1}{2}\frac{d\rho^2}{dt}$  on the leftmost and rightmost sides, we have

$$
- \frac {d \mathcal {L}}{d t} \cdot \left(\frac {1}{2} \frac {d \rho^ {2}}{d t}\right) ^ {- 1} = \frac {d}{d t} \log \rho + \left(\frac {d}{d t} \log \rho\right) ^ {- 1} \left\| \frac {d \hat {\pmb {\theta}}}{d t} \right\| _ {2} ^ {2}.
$$

By  $-\frac{d\mathcal{L}}{dt} \geq 0$  and (7), the LHS is no greater than  $-\frac{d\mathcal{L}}{dt} \cdot \left(L \cdot \mathcal{L}\log \frac{1}{\mathcal{L}}\right)^{-1} = \frac{1}{L}\log \log \frac{1}{\mathcal{L}}$ . Thus we have  $\frac{d}{dt}\log \log \frac{1}{\mathcal{L}} - L\frac{d}{dt}\log \rho \geq L\left(\frac{d}{dt}\log \rho\right)^{-1}\left\| \frac{d\hat{\boldsymbol{\theta}}}{dt}\right\|_2^2$ , where the LHS is exactly  $\frac{d}{dt}\log \tilde{\gamma}$ .

# 6 DISCUSSION AND FUTURE DIRECTIONS

In this paper, we analyze the dynamics of gradient flow/descent of homogeneous neural networks under a minimal set of assumptions. The main technical contribution of our work is to prove rigorously that for gradient flow/descent, the normalized margin is increasing and converges to a KKT point of a natural max-margin problem. Our results lead to some natural further questions:

- Can we generalize our results for gradient descent on smooth neural networks to non-smooth ones? In the smooth case, we can lower bound the decrement of training loss by the gradient norm squared, multiplied by a factor related to learning rate. However, in the non-smooth case, no such inequality is known in the optimization literature, and it is unclear what kind of natural assumption can make it holds.  
- Can we make more structural assumptions on the neural network to prove stronger results? In this work, we use a minimal set of assumptions to show that the convergent direction of parameters is a KKT point. A potential research direction is to identify more key properties of modern neural networks and show that the normalized margin at convergence is locally or globally optimal (in terms of optimizing (P)).  
- Can we extend our results to neural networks with bias terms? In our experiments, the normalized margin of the CNN with bias also increases during training despite that its output is non-homogeneous. It is very interesting (and technically challenging) to provide a rigorous proof for this fact.

# REFERENCES

Pierre-Antoine Absil, Robert Mahony, and Benjamin Andrews. Convergence of the iterates of descent methods for analytic cost functions. SIAM Journal on Optimization, 16(2):531-547, 2005.  
Alnur Ali, J Zico Kolter, and Ryan J Tibshirani. A continuous-time view of early stopping for least squares regression. arXiv preprint arXiv:1810.10082, 2018.  
Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via overparameterization. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 242-252, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. arXiv preprint arXiv:1905.13655, 2019a.

Sanjeev Arora, Simon S Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. arXiv preprint arXiv:1904.11955, 2019b.  
Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 274-283, Stockholm, Sweden, 10-15 Jul 2018. PMLR.  
Andrzej Banburski, Qianli Liao, Brando Miranda, Tomaso Poggio, Lorenzo Rosasco, and Jack Hidary. Theory III: Dynamics and generalization in deep networks. CBMM Memo No: 090, version 20, 2019.  
Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 6240-6249. Curran Associates, Inc., 2017.  
Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić, Pavel Laskov, Giorgio Giacinto, and Fabio Roli. Evasion attacks against machine learning at test time. In Hendrik Blockeel, Kristian Kersting, Siegfried Nijssen, and Filip Železný, editors, Machine Learning and Knowledge Discovery in Databases, pages 387-402, Berlin, Heidelberg, 2013. Springer Berlin Heidelberg.  
Guy Blanc, Neha Gupta, Gregory Valiant, and Paul Valiant. Implicit regularization for deep neural networks driven by an Ornstein-uhlenbeck like process. arXiv preprint arXiv:1904.09080, 2019.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pages 39-57, May 2017. doi: 10.1109/SP.2017.49.  
Francis H. Clarke, Yuri S. Ledyaev, Ronald J. Stern, and Peter R. Wolenski. Nonsmooth analysis and control theory, volume 178. Springer Science & Business Media, 2008.  
Frank H. Clarke. Generalized gradients and applications. Transactions of the American Mathematical Society, 205:247-262, 1975.  
Frank H Clarke. Optimization and Nonsmooth Analysis. Society for Industrial and Applied Mathematics, 1990. doi: 10.1137/1.9781611971309.  
Michel Coste. An Introduction to  $O$ -minimal Geometry. 2002.  
Haskell B Curry. The method of steepest descent for non-linear minimization problems. Quarterly of Applied Mathematics, 2(3):258-261, 1944.  
Damek Davis, Dmitriy Drusvyatskiy, Sham Kakade, and Jason D. Lee. Stochastic subgradient method converges on tame functions. Foundations of Computational Mathematics, Jan 2019. ISSN 1615-3383. doi: 10.1007/s10208-018-09409-5.  
Dmitriy Drusvyatskiy, Alexander D Ioffe, and Adrian S Lewis. Curves of descent. SIAM Journal on Control and Optimization, 53(1):114-138, 2015.  
Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 1675-1685, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Simon S Du and Jason D Lee. On the power of over-parametrization in neural networks with quadratic activation. arXiv preprint arXiv:1803.01206, 2018.  
Simon S. Du, Wei Hu, and Jason D. Lee. Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 382-393. Curran Associates, Inc., 2018.

Joydeep Dutta, Kalyanmoy Deb, Rupesh Tulshyan, and Ramnik Arora. Approximate KKT points and a proximity measure for termination. Journal of Global Optimization, 56(4):1463-1499, 2013.  
Gauthier Gidel, Francis Bach, and Simon Lacoste-Julien. Implicit regularization of discrete gradient dynamics in deep linear neural networks. arXiv preprint arXiv:1904.13262, 2019.  
Giorgio Giorgi, Angelo Guerraggio, and Jörg Thierfelder. Chapter IV - Nonsmooth Optimization Problems. In Mathematics of Optimization, pages 359 - 457. Elsevier Science, Amsterdam, 2004. ISBN 978-0-444-50550-7.  
Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In Sébastien Bubeck, Vianney Perchet, and Philippe Rigollet, editors, Proceedings of the 31st Conference On Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 297-299. PMLR, 06-09 Jul 2018.  
Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias in terms of optimization geometry. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1832-1841, Stockholmssan, Stockholm Sweden, 10-15 Jul 2018a. PMLR.  
Suriya Gunasekar, Jason D Lee, Daniel Soudry, and Nati Srebro. Implicit bias of gradient descent on linear convolutional networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 9482-9491. Curran Associates, Inc., 2018b.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In The IEEE International Conference on Computer Vision (ICCV), December 2015.  
Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 8571-8580. Curran Associates, Inc., 2018.  
Ziwei Ji and Matus Telgarsky. Risk and parameter convergence of logistic regression. arXiv preprint arXiv:1803.07300, 2018.  
Ziwei Ji and Matus Telgarsky. Gradient descent aligns the layers of deep linear networks. In International Conference on Learning Representations, 2019a.  
Ziwei Ji and Matus Telgarsky. A refined primal-dual analysis of the implicit bias. arXiv preprint arXiv:1906.04540, 2019b.  
Jason M Klusowski and Andrew R Barron. Approximation by combinations of relu and squared relu ridge functions with 11 and 10 controls. IEEE Transactions on Information Theory, 64(12): 7649-7656, 2018.  
Jaehoon Lee, Lechao Xiao, Samuel S Schoenholz, Yasaman Bahri, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. arXiv preprint arXiv:1902.06720, 2019.  
Bo Li, Shanshan Tang, and Hajjun Yu. Better approximations of high dimensional smooth functions by deep neural networks with rectified power units. arXiv preprint arXiv:1903.05858, 2019.  
Xingguo Li, Junwei Lu, Zhaoran Wang, Jarvis Haupt, and Tuo Zhao. On tighter generalization bound for deep neural networks: Cnns, resnets, and beyond. arXiv preprint arXiv:1806.05159, 2018a.  
Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. Algorithmic regularization in over-parameterized matrix sensing and neural networks with quadratic activations. In Conference On Learning Theory, pages 2-47, 2018b.

Cong Ma, Kaizheng Wang, Yuejie Chi, and Yuxin Chen. Implicit regularization in nonconvex statistical estimation: Gradient descent converges linearly for phase retrieval, matrix completion and blind deconvolution. arXiv preprint arXiv:1711.10467, 2017.  
Mor Shpigel Nacson, Nathan Srebro, and Daniel Soudry. Stochastic gradient descent on separable data: Exact convergence with a fixed learning rate. arXiv preprint arXiv:1806.01796, 2018.  
Mor Shpigel Nacson, Suriya Gunasekar, Jason Lee, Nathan Srebro, and Daniel Soudry. Lexicographic and depth-sensitive margins in homogeneous and non-homogeneous deep models. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 4683-4692, Long Beach, California, USA, 09-15 Jun 2019a. PMLR.  
Mor Shpigel Nacson, Jason Lee, Suriya Gunasekar, Pedro Henrique Pamplona Savarese, Nathan Srebro, and Daniel Soudry. Convergence of gradient descent on separable data. In The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan, pages 3420-3428, 2019b.  
Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. In search of the real inductive bias: On the role of implicit regularization in deep learning. arXiv preprint arXiv:1412.6614, 2014.  
Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-SGD: Path-normalized optimization in deep neural networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2422-2430. Curran Associates, Inc., 2015.  
Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro. A PAC-bayesian approach to spectrally-normalized margin bounds for neural networks. In International Conference on Learning Representations, 2018.  
J Jr Palis and Welington De Melo. Geometric theory of dynamical systems: an introduction. Springer Science & Business Media, 2012.  
Aaditya Ramdas and Javier Pena. Towards a deeper geometric, analytic and algorithmic understanding of margins. Optimization Methods and Software, 31(2):377-391, 2016.  
Saharon Rosset, Ji Zhu, and Trevor J. Hastie. Margin maximizing loss functions. In S. Thrun, L. K. Saul, and B. Schölkopf, editors, Advances in Neural Information Processing Systems 16, pages 1237-1244. MIT Press, 2004.  
Robert E. Schapire and Yoav Freund. Boosting: Foundations and Algorithms. The MIT Press, 2012. ISBN 0262017180, 9780262017183.  
Robert E Schapire, Yoav Freund, Peter Bartlett, Wee Sun Lee, et al. Boosting the margin: A new explanation for the effectiveness of voting methods. The annals of statistics, 26(5):1651-1686, 1998.  
Shai Shalev-Shwartz and Yoram Singer. On the equivalence of weak learnability and linear separability: New relaxations and efficient boosting algorithms. Machine learning, 80(2-3):141-163, 2010.  
Jure Sokolic, Raja Giryes, Guillermo Sapiro, and Miguel R. D. Rodrigues. Robust large margin deep neural networks. IEEE Trans. Signal Processing, 65(16):4265-4280, 2017. doi: 10.1109/TSP.2017.2708039.  
Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. Journal of Machine Learning Research, 19, 2018a.  
Daniel Soudry, Elad Hoffer, and Nathan Srebro. The implicit bias of gradient descent on separable data. In International Conference on Learning Representations, 2018b.

Arun Suggala, Adarsh Prasad, and Pradeep K Ravikumar. Connecting optimization and regularization paths. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 10608-10619. Curran Associates, Inc., 2018.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2013.  
Matus Telgarsky. Margins, shrinkage, and boosting. In Sanjoy Dasgupta and David McAllester, editors, Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 307-315, Atlanta, Georgia, USA, 17-19 Jun 2013. PMLR.  
Lou van den Dries and Chris Miller. Geometric categories and o-minimal structures. Duke Mathematical Journal, 84(2):497-540, 1996.  
Colin Wei, Jason D Lee, Qiang Liu, and Tengyu Ma. Regularization matters: Generalization and optimization of neural nets v.s. their induced kernel. arXiv preprint arXiv:1810.05369, 2018.  
Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 4148-4158. Curran Associates, Inc., 2017.  
Tengyu Xu, Yi Zhou, Kaiyi Ji, and Yingbin Liang. When will gradient methods converge to max-margin classifier under relu models?, 2018.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations, 2017.  
Hongyi Zhang, Yann N. Dauphin, and Tengyu Ma. Fixup initialization: Residual learning without normalization. In International Conference on Learning Representations, 2019.  
Kai Zhong, Zhao Song, Prateek Jain, Peter L Bartlett, and Inderjit S Dhillon. Recovery guarantees for one-hidden-layer neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 4140-4149. JMLR.org, 2017.  
Difan Zou, Yuan Cao, Dongruo Zhou, and Quanquan Gu. Stochastic gradient descent optimizes over-parameterized deep relu networks, 2018.  
Guus Zoutendijk. Mathematical programming methods. 1976.
