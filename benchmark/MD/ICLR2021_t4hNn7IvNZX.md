# CERTIFIED DISTRIBUTIONAL ROBUSTNESS VIA SMOOTHED CLASSIFIERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The robustness of deep neural networks against adversarial example attacks has received much attention recently. We focus on certified robustness of smoothed classifiers in this work, and propose to use the worst-case population loss over noisy inputs as a robustness metric. Under this metric, we provide a tractable upper bound serving as a robustness certificate by exploiting the duality. To improve the robustness, we further propose a noisy adversarial learning procedure to minimize the upper bound following the robust optimization framework. The smoothness of the loss function ensures the problem easy to optimize even for non-smooth neural networks. We show how our robustness certificate compares with others and the improvement over previous works. Experiments on a variety of datasets and models verify that in terms of empirical accuracies, our approach exceeds the state-of-the-art certified/heuristic methods in defending adversarial examples.

# 1 INTRODUCTION

Deep neural networks (DNNs) have been known to be vulnerable to adversarial example attacks: by feeding the DNN with slightly perturbed inputs, the attack alters the prediction output. The attack can be fatal in performance-critical systems such as autonomous vehicles or automated tumor diagnosis. A DNN is robust when it can resist such an attack that, as long as the range of the perturbation is not too large (usually invisible by human), the model produces an expected output despite of the specific perturbation. Various approaches have been proposed for improving the robustness of DNNs, with or without a performance guarantee.

Although a number of approaches have been proposed for certified robustness, it is vague how robustness should be defined. For example, works including Cohen et al. (2019); Pinot et al. (2019); Li et al. (2019); Lecuyer et al. (2019) propose smoothed classifiers to ensure the inputs with adversarial perturbation to be classified into the same class as the inputs without. However, since both inputs are inserted randomly noise, it cannot be guaranteed that the inputs are classified into the correct class. It is possible that the adversarially perturbed input has the same label as the original one which is wrongly classified by the DNN. In this case, the robustness guarantee does not make sense any more. Further, the robustness guarantee is provided at the instance level, i.e., within a certain perturbation range, the modification of an input instance cannot affect the prediction output. But a DNN is a statistical model to be evaluated on the input distribution, rather than a single instance. Instead of counting the number of input instances meeting the robustness definition, it is desired to evaluate the robustness of a DNN over the input distribution.

We introduce the distributional risk as a DNN robustness metric, and propose a noisy adversarial learning (NAL) procedure based on distributional robust optimization, which provides a provable guarantee. Assume a base classifier  $f$  trying to map instance  $x_0$  to corresponding label  $y$ . It is found that when fed with the perturbed instance  $x$  (within a  $l_2$  ball centered at  $x_0$ ), a smoothed classifier  $g(x) = \mathbb{E}_z[f(x + z)]$  with  $z \sim \mathcal{N}(0, \sigma^2 I)$  can provably return the same label as  $g(x_0)$  does. However, we think such a robustness guarantee cannot ensure  $g(x_0)$  to be correctly classified as  $y$ , resulting in unsatisfying performance in practice. Instead, we evaluate robustness as the worst-case loss over the distribution of noisy inputs. For simplicity, we jointly express the input instance and the label as  $x_0 \sim P_0$  where  $P_0$  is the distribution of the original input. By using  $\ell(\cdot)$  as the loss function, we evaluate DNNs by the worst-case distributional risk:  $\sup_P \mathbb{E}_P[\ell(\theta; x + z)]$ . The classifier is parameterized by  $\theta \in \Theta$ , and  $x + z \sim P$  where  $P$  is a distribution within a certain

distance from  $P_0$ . We prove such a loss is upper bounded by a data-dependent certificate, which can be optimized by the noisy adversarial training procedure:

$$
\underset {\theta \in \Theta} {\text {m i n i m i z e}} \sup  _ {P \in \mathcal {P}} \mathbb {E} _ {P} [ \ell (\theta ; x + z) ]. \tag {1}
$$

Compared to previous robustness certificates via smoothed classifiers, our method provides a provable guarantee w.r.t. the ground truth input distribution. Letting the optimized  $\theta$  be the parameter of  $g(\cdot)$  and  $f(\cdot)$  respectively, we further show that the smoothed classifier  $g(\cdot)$  provides an improved robustness certificate than that of  $f(\cdot)$ , due to a tighter bound on the worst-case loss.

The key is that, for mild perturbations, we adopt a Lagrangian relaxation for the usual loss  $\ell (\theta ;x + z)$  as the robust surrogate, and the surrogate is strongly concave in  $x$  and hence easy to optimize. Our approach enjoys convergence guarantee similar to the method in Sinha et al. (2018), but different from Sinha et al. (2018), our approach does not require  $\ell$  to be smooth, and thus can be applied to arbitrary neural networks. The advantage of the smoothed classifier also lies in a tighter robustness certificate than the base classifier. The intuition is that, in the inner maximization step, instead of seeking one direction which maximizes the loss, our approach performs gradient ascent along the direction which maximizes the total loss of examples sampled from the neighborhood of the original input. The noisy adversarial training procedure produces smoothed classifiers robust against the neighborhood of the worst-case adversarial examples with a certified bound.

Highlights of our contribution are as follows. First, we review the drawbacks in the previous definition of robustness, and propose to evaluate robustness by the worst-case loss over the input distribution. Second, we derive a data-dependent upper bound for the worst-case loss, constituting a robustness certificate. Third, by minimizing the robustness certificate in the training loop, we propose noisy adversarial learning for enhancing model robustness, in which the smoothness property entails the computational tractability of the certificate. Through both theoretical analysis and experimental results, we verify that our certified DNNs enjoy better accuracies compared with the state-of-the-art defending adversarial example attacks.

# 2 RELATED WORK

Works proposed to defend against adversarial example attacks can be categorized into the following categories.

In empirical defences, there is no guarantee how the DNN model would perform against the adversarial examples. Stability training (Zheng et al. (2016); Zantedeschi et al. (2017); Liu et al. (2018)) improves model robustness by adding randomized noise to the input during training but shows limited performance enhancement. Adversarial training (Goodfellow et al. (2015); Kurakin et al. (2018); Madry et al. (2017); Kannan et al. (2018); Zhang et al. (2019); He et al. (2019); Wang et al. (2019)) trains over adversarial examples found at each training step but unfortunately does not guarantee the performance over unseen adversarial inputs. Although without a guarantee, adversarial training has excellent performance in empirical defences against adversarial attacks.

Certified defences are certifiably robust against any adversarial input within an  $\ell_p$ -norm perturbation range from the original input. A line of works construct a computationally tractable relaxation for computing an upper bound on the worst-case loss over all valid attacks. The relaxations include linear programming (Wong & Kolter (2018)), mixed integer programming (Tjeng et al. (2018)), semidefinite programming (Raghunathan et al. (2018)), and convex relaxation (Namkoong & Duchi (2017); Salman et al. (2019b)). But those deterministic methods are not scalable. Some works such as Dvijotham et al. (2018) formulate the search for the largest perturbation range as an optimization problem and solve its dual problem. Sinha et al. (2018) also propose a robustness certificate based on a Lagrangian relaxation of the loss function, and it is provably robust against adversarial input distributions within a Wasserstein ball centered around the original input distribution. The certificate of our work is constructed on a Lagrangian relaxation form of the worst-case loss, but has a broader applicability than Sinha et al. (2018) with a tighter loss bound due to the smoothness property.

An alternative line of works propose to select appropriate surrogates for each neuron activation layer by layer (Weng et al. (2018); Zhang et al. (2018)) to facilitate the search for a certified lower bound. By integrating with interval bound propagation (Gowal et al. (2018)), Zhang et al. (2020) make the search computationally efficient and scalable. Other works (Mirman et al. (2018); Singh

et al. (2018)) apply the abstract interpretation to train provably robust neural networks. Our work is orthogonal to these works.

Randomized smoothing introduces randomized noise to the neural network, and tries to provide a statistically certified robustness guarantee. Pinot et al. (2020) have demonstrated by game theory that no deterministic classifier can claim to be more robust than all others against any possible adversarial attack. But such a question remains open in the randomized regime, where randomized smoothing can be considered as a contributing effort. The smoothing method does not depend on a specific neural network, or a type of relaxation, but can be generally applied to arbitrary neural networks. The idea of adding randomized noise was first proposed by Lecuyer et al. (2019), given the inspiration of the differential privacy property, and then Li et al. (2019) improve the certificate with Rényi divergence. Cohen et al. (2019) obtain a larger certified robustness bound through the smoothed classifier based on Neyman-Pearson theorem. Phan et al. (2020) extend the noise addition mechanism to large-scale parallel algorithms. By extending the randomized noise to the general family of exponential distributions, Pinot et al. (2019) unify previous approaches to preserve robustness to adversarial attacks. Lee et al. (2019) offer adversarial robustness guarantees for  $\ell_0$ -norm attacks. Both Salman et al. (2019a); Jia et al. (2019) employ adversarial training to improve the performance of randomized smoothing. Following a similar principle, our work trains over adversarial data with randomized noise. But we provide a more practical robustness certificate and a training method achieving higher empirical accuracy than theirs.

# 3 PROPOSED APPROACH

We first define the closeness between distributions, based on which we constrain how far the input distribution is perturbed. Then we introduce our definition of robustness on smoothed classifiers. Our main theorem gives a tractable robustness certificate which is easy to optimize. Our algorithm for improving the robustness of the smoothed classifiers is provided. All proofs are collected in the appendices for conciseness.

# 3.1 A DISTRIBUTIONAL ROBUSTNESS CERTIFICATE

Definition 1 (Wasserstein distance). Wasserstein distances define a notion of closeness between distributions. Let  $(\mathcal{X} \subset \mathbb{R}^d, \mathcal{A}, P)$  be a probability space and the transportation cost  $c: \mathcal{X} \times \mathcal{X} \to [0, \infty)$  be nonnegative, lower semi-continuous, and  $c(x, x) = 0$ .  $P$  and  $Q$  are two probability measures supported on  $\mathcal{X}$ . Let  $\Pi(P, Q)$  denotes the collection of all measures on  $\mathcal{X} \times \mathcal{X}$  with marginals  $P$  and  $Q$  on the first and second factors respectively, i.e., it holds that  $\pi(A, \mathcal{X}) = P(A)$  and  $\pi(\mathcal{X}, A) = Q(A)$ ,  $\forall A \in \mathcal{A}$ . The Wasserstein distance between  $P$  and  $Q$  is

$$
W _ {c} (P, Q) := \inf  _ {\pi \in \Pi (P, Q)} \mathbb {E} _ {\pi} [ c (x, y) ]. \tag {2}
$$

For example, the  $\ell_2$ -norm  $c(x,x_0) = \| x - x_0\| _2^2$  satisfies the aforementioned conditions.

Distributional robustness. Assume the original input  $x_0$  is drawn from the distribution  $P_0$ , and the perturbed input is  $x$ . Each input is added randomized Gaussian noise  $z \sim \mathcal{N}(0, \sigma^2 I)$  before being fed to the classifier. Instead of regarding the noise as a part of the smoothed classifier, we treat  $\hat{x} = x + z$  as a noisy input coming from the distribution  $P$  in the analysis. Since  $z \in \mathbb{R}^d$ , we need to set  $\mathcal{X} = \mathbb{R}^d$  to admit  $\hat{x} \in \mathcal{X}$  as Lecuyer et al. (2019); Cohen et al. (2019); Salman et al. (2019a) do. Since the perturbed input should be visually indistinguishable from the original one, we define the robustness region as  $\mathcal{P} = \{P : W_c(P, P_0) \leq \rho, P \in P(\mathcal{X})\}$ , where  $\rho > 0$ . Within such a region, we evaluate the robustness as a worst-case population loss over noisy inputs:  $\sup_{P \in \mathcal{P}} \mathbb{E}_P[\ell(\theta; \hat{x})]$ . Essentially, we evaluate the robustness of a smoothed classifier based on its performance on the worst-case adversarial example distribution. A smaller loss indicates a higher level of robustness. We will compare the definition against others in the next section. However, such a robustness metric is impossible to measure in practice as we have no idea about  $P$ . Even if  $P$  can be acquired, it can be a non-convex region which renders the constrained optimization objective intractable. Hence we resort to the Lagrangian relaxation of the problem by assuming a dual variable  $\gamma$ .

As the main theorem of this work, we provide an upper bound for the worst-case population loss for any level of robustness  $\rho$ . We further show that for small enough  $\rho$ , the upper bound is tractable and easy to optimize.

Theorem 1. Let  $\ell : \Theta \times \mathcal{X} \to \mathbb{R}$  and transportation cost function  $c: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_+$  be continuous. Let  $x_0 \sim P_0$  be the input,  $x$  be the adversarial example and  $z \sim \mathcal{N}(0, \sigma^2 I)$  be the additive noise of the same shape as  $x$ . We assume  $\hat{x} = x + z \sim P \in \mathcal{P}$  and  $\phi_{\gamma}(\theta; x_0) = \mathbb{E}_z \sup_{x \in \mathcal{X}} \{\ell(\theta; x + z) - \gamma c(x + z, x_0)\}$  be the robust surrogate. For any  $\gamma, \rho > 0$ , we have

$$
\sup  _ {P \in \mathcal {P}} \mathbb {E} _ {P} [ \ell (\theta ; \hat {x}) ] \leq \gamma \rho + \mathbb {E} _ {P _ {0}} \left[ \phi_ {\gamma} (\theta ; x _ {0}) \right]. \tag {3}
$$

The proof is given in Appendix A.1. It is notable that the right-hand side takes the expectation over  $P_0$  and  $z$  respectively, and given a particular input  $x_0$  and a noise sample  $z$ , we seek an adversarial example which maximizes the surrogate loss. Typically,  $P_0$  is impossible to obtain and thus we use an empirical distribution, such as the training data distribution, to approximate  $P_0$  in practice.

Since Thm. 1 provides an upper bound for the worst-case population loss, it offers a principled adversarial training approach which minimizes the upper bound instead of the actual loss, i.e.,

$$
\underset {\theta \in \Theta} {\text {m i n i m i z e}} \mathbb {E} _ {P _ {0}} \left[ \phi_ {\gamma} \left(\theta ; x _ {0}\right) \right]. \tag {4}
$$

In the following we show the above loss function has a form which is tractable for arbitrary neural networks, due to a smoothed loss function. Hence Thm. 1 provides a tractable robustness certificate depending on the data.

Properties of the smoothed classifier. We show the optimization objective of Eq. 4 has a form which is tractable for any neural network, particular for the non-smooth ones with ReLU activation layers. More importantly, the smoothness of the classifier enables the adversarial training procedure to converge as we want by using the common optimization techniques such as stochastic gradient descent. The smoothness of the loss function comes from the smoothed classifier with randomized noise. Specifically,

Theorem 2. Assume  $\ell : \Theta \times \mathcal{X} \to [0, M]$  is a bounded loss function. The loss function on the smoothed classifier can be expressed as  $\hat{\ell}(\theta; x) \coloneqq \mathbb{E}_z[\ell(\theta; x + z)]$ ,  $z \sim \mathcal{N}(0, \sigma^2 I)$ . Then we have  $\hat{\ell}$  is  $2M$ -smooth w.r.t.  $\ell_2$ -norm, i.e.,  $\hat{\ell}$  satisfies

$$
\left\| \nabla_ {x} \hat {\ell} (\theta ; x) - \nabla_ {x} \hat {\ell} \left(\theta ; x ^ {\prime}\right) \right\| _ {2} \leq 2 M \| x - x ^ {\prime} \| _ {2}. \tag {5}
$$

The proof is in Appendix A.2. It mainly takes advantage of the randomized noise which has a smoothing effect on the loss function. For DNNs with non-smooth layers, the smoothed classifier makes it up and turns the loss function to a smoothed one, which contributes as an important property to the strong concavity of  $\mathbb{E}_z\left\{\ell (\theta ;x + z) - \gamma c(x + z,x_0)\right\}$  and therefore ensures the tractability of the robustness certificate.

Corollary 1. For any  $c: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_+ \cup \{\infty\}$  1-strongly convex in its first argument, and  $\hat{\ell}: x \mapsto \mathbb{E}_z[\ell(\theta; x + z)]$  being  $2M$ -smooth, the function  $\mathbb{E}_z\left\{\ell(\theta; x + z) - \gamma c(x + z, x_0)\right\}$  is strongly concave in  $x$  for any  $\gamma \geq 2M$ .

The proof is in Appendix A.3. Note that here we specify the requirement on the transportation cost  $c$  to be 1-strongly convex in its first argument. The  $\ell_2$ -norm cost satisfies the condition. Before showing how the strong concavity plays a part in the convergence, we illustrate our algorithm first.

# 3.2 NOISY ADVERSARIAL LEARNING ALOGRITHM

Problem 4 provides an explicit way to improve the robustness of a smoothed classifier parameterized by  $\theta$ . We correspondingly design a noisy adversarial learning algorithm to obtain the classifier of which its robustness can be guaranteed. In the algorithm, we use the empirical distribution to replace the ideal input distribution  $P_0$ , and sample  $z$  a number of times to substitute the expectation with the sample average. Assuming we have a total of  $n$  training instances  $x_0^i, \forall i \in [n]$ , and sample  $z_{ij} \sim \mathcal{N}(0, \sigma^2 I)$  for the  $i$ -th instance for  $s$  times, the objective is:

$$
\underset {\theta \in \Theta} {\text {m i n i m i z e}} \frac {1}{n s} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {s} \sup  _ {x \in \mathcal {X}} \left[ \ell (\theta ; x + z _ {i j}) - \gamma c \left(x + z _ {i j}, x _ {0} ^ {i}\right) \right]. \tag {6}
$$

The detail of the algorithm is illustrated in Alg. 1. In the inner maximization step (line 3-6), we adopt the projected gradient descent (PGD Madry et al. (2017); Kurakin et al. (2018)) to approximate the maximizer according to the convention. The hyperparameters include the number of iterations  $K$  and the learning rate  $\eta_1$ . Within each iteration, we sample the Gaussian noise  $s$  times, given which we compute an average perturbation direction for each update. The more noise samples, the closer the averaging result is to the expectation value, which is definitely at the sacrifice of higher computation expense. Similarly, a larger number of  $K$  indicates stronger adversarial attacks and higher model robustness, but also incurs higher computation complexity. Hence choosing appropriate values of  $s$  and  $K$  is important in practice.

Algorithm 1 Training Phase of NAL  
Input: batch size  $n$  , number of noise samples  $s$  , noise STD  $\sigma$  , learning rate  $\eta_{1},\eta_{2}$  , number of iterations  $K$  , penalty parameter  $\gamma$  , training iterations  $T$    
Output: the classifier parameter  $\theta$    
1: for  $t\in \{1,\dots,T\}$  do   
2: for  $i\in \{1,\ldots ,n\}$  do   
3: for  $k\in \{0,\dots,K - 1\}$  do   
4:  $\Delta x_k^i = \frac{1}{s}\sum_{j = 1}^{s}\nabla_{x_k^i}\ell (\theta ;x_k^i +z_{ij}) - \gamma \nabla_{x_k^i}c(x_k^i +z_{ij},x_0^i)$  , where  $z_{ij}\sim \mathcal{N}(0,\sigma^2 I)$    
5:  $x_{k + 1}^{i} = x_{k}^{i} + \eta_{1}\Delta x_{k}^{i}$    
6: end for   
7: end for   
8:  $\theta^{t + 1} = \theta^t -\eta_2\left\{\frac{1}{ns}\sum_{i = 1}^{n}\left[\nabla_\theta \sum_{j = 1}^{s}\ell (\theta^t;x_K^i +z_{ij})\right]\right\}$    
9: end for

After training is done, we obtain the classifier parameter  $\theta$ . In the inference phase, we sample a number of  $z \sim \mathcal{N}(0, \sigma^2 I)$  to add to the testing instance. The noisy testing examples are fed to the classifier to get the prediction outputs.

Convergence. An important property associated with the smoothed classifier is the strong concavity of the robust surrogate loss, which is the key to the convergence proof. The detail of the proof can be found in Appendix A.4. As long as the loss  $\hat{\ell}$  is smooth on the parameter space  $\Theta$ , NAL has a convergence rate  $O(1 / \sqrt{T})$ , similar to Sinha et al. (2018), but NAL does not need to replace the non-smooth layer ReLU with Sigmoid or ELU to guarantee robustness.

# 4 A TIGHTER BOUND

We compare our work with the state-of-the-art robustness definitions and certificates in this section.

# 4.1 ADVERSARIAL TRAINING

Our approach improves the distributional robustness certificate proposed by Sinha et al. (2018). In Sinha et al. (2018), a classifier  $f$  maps input instance  $x_0 \sim P_0$  to corresponding label  $y$ . They perturb  $x_0$  to  $x'$  in the same robustness region as ours:  $\mathcal{P} = \{P : W_c(P, P_0) \leq \rho, P \in P(\mathcal{X})\}$ , where  $\rho > 0$ . But their worst-case population loss is defined on the base classifier without noise:  $\sup_{x' \sim P: W_c(P, P_0) \leq \rho} \mathbb{E}_P[\ell(\theta; x')]$ . We show that, given the same classifier parameter  $\theta$ , our worst-case loss is smaller than Sinha et al. (2018), suggesting a better robustness certificate.

Theorem 3. Under the same denotations and conditions as Thm. 1, we have

$$
\begin{array}{l} \sup  _ {\hat {x} \sim P: W _ {c} (P, P _ {0}) \leq \rho} \mathbb {E} _ {P} [ \ell (\theta ; \hat {x}) ] \leq \inf  _ {\gamma \geq 0} \left\{\gamma \rho + \mathbb {E} _ {P _ {0}} \left[ \phi_ {\gamma} (\theta ; x _ {0}) \right] \right\} \\ \leq \inf  _ {\gamma \geq 0} \left\{\gamma \rho + \mathbb {E} _ {P _ {0}} \sup  _ {x ^ {\prime} \in \mathcal {X}} [ \ell (\theta ; x ^ {\prime}) - \gamma c (x ^ {\prime}, x _ {0}) ] \right\} = \sup  _ {x ^ {\prime} \sim P: W _ {c} (P, P _ {0}) \leq \rho} \mathbb {E} _ {P} [ \ell (\theta ; x ^ {\prime}) ]. \tag {7} \\ \end{array}
$$

The proof is given in Appendix A.5. We demonstrate that not only the worst-case loss is smaller, but the tractable upper bound is smaller than the certificate of Sinha et al. (2018). If the outer minimiza

tion problem applies to both sides of the inequality, our approach would obtain a smaller loss when both classifiers share the same neural architecture.

# 4.2 SMOOTHED CLASSIFIERS

Works including Lecuyer et al. (2019); Cohen et al. (2019); Pinot et al. (2019); Li et al. (2019) and others guarantee the robustness of a DNN classifier by inserting randomized noise to the input at the inference phase. Most of them do not concern about the training phase, but merely provide a deterministic relationship between the robustness certificate and the additive noise. Specifically, we have the original input  $x_0 \in \mathcal{X}$  and its perturbation  $x$  within a given range  $\| x - x_0 \|_2 \leq \varepsilon$ . The smoothed classifier  $g(x)$  returns class  $c_i$  with probability  $p_i$ . For instance  $x_0$ , robustness is defined by the largest perturbation radius  $R$  which does not alter the instance's prediction, i.e.,  $g(x)$  is classified into the same category as  $g(x_0)$ . Such perturbation radius depends on the largest and second largest probabilities of  $p_i$ , denoted by  $p_A, p_B$  respectively. For example, the results in Cohen et al. (2019) have shown that  $R = \frac{\sigma}{2} \left( \Phi^{-1} \left( \underline{p_A} \right) - \Phi^{-1} \left( \overline{p_B} \right) \right)$  where  $\Phi^{-1}$  is the inverse of the standard Gaussian CDF,  $p_A$  is a lower bound of  $p_A$ , and  $\overline{p_B}$  is an upper bound of  $p_B$ .

The previous robustness definition only guarantees  $g(x)$  to be classified to the same class as  $g(x_0)$ , but ignores the fact that  $g(x_0)$  may be wrongly classified, which is not a precise definition. To make up for it, Li et al. (2019) propose stability training with noise (STN) and Cohen et al. (2019) adopt training with noise, both of which learn smoothed classifiers mapping noisy inputs to correct labels. However, there is no guarantee to ensure  $g(x_0)$  to be correctly labeled. Actually we found the robustness mainly comes from the STN/training with noise, rather than the noise addition at the inference. In Fig. 1, we could observe that the model performance indeed improves when tested with noise. However, the classifier trained without additive noise (triangle) degrades significantly compared with STN/training with noise (diamond/circle). The result is an evidence that a classifier almost cannot defend adversarial attacks when trained without but tested with additive noise. Therefore, we conclude the smoothed classifier can only improve robustness only if the base classifier is robust.

We consider robustness refers to the ability of a DNN to classify adversarial examples into the correct classes, and such an ability should be evaluated on the population of adversarial examples, not a single instance. For a fair comparison, we rewrite the certificate in Cohen

et al. (2019) as an expression on single-instance losses in Appendix A.6, and show that such a condition does not reflect how the DNN performs on the majority of data instances.

![](images/37a61e0a6b037b477097a81962d4b0361a48f58ff777f44a772f457d581fc510.jpg)  
Figure 1: Accuracies of models trained on MNIST under different levels of  $\ell_2$  attacks. Undefend means a naturally trained model. Solid lines represent models tested with additive noise, and dotted lines mean that without.  $\sigma = 0.1$  means adding Gaussian noise  $\mathcal{N}(0,0.1^2\mathbf{I})$

# 5 EXPERIMENT

Baselines, datasets and models. Testing accuracies under different levels of adversarial attacks are chosen as the metric. We compare the empirical performance of NAL with representative baselines including: WRM (Sinha et al. (2018)), SmoothAdv (Salman et al. (2019a)), STN (Li et al. (2019)) and TRADES (Zhang et al. (2019)). Since WRM requires the loss function to be smooth, we follow the convention to adapt the ReLU activation layer to the ELU layer. SmoothAdv combines adversarial training with the smoothed classifier and claims to be superior than Cohen et al. (2019). Hence we omit Cohen et al. (2019) in comparison. TRADES is an adversarial training algorithm which won 1st place in the NeurIPS 2018 Adversarial Vision Challenge. Experiments are conducted on datasets MNIST, CIFAR-10, and Tiny ImageNet, and models including a three-layer CNN, ResNet-18, VGG-16, and their corresponding variants with ReLu replaced by ELU for fair comparison with WRM. The cross-entropy loss is chosen for  $\ell$  and  $c(x,x_0) = \| x - x_0\| _2^2$  is selected as the cost function.

Training hyperparameters. Table 1 gives the training hyperparameters in NAL and the batch size is chosen as 128. The hyperparameters used in baselines are supplied in Appendix B.1. Since NAL

<table><tr><td>Dataset</td><td>η1</td><td>η2</td><td>epochs</td><td>σ</td><td>γ</td><td>ε</td></tr><tr><td>MNIST</td><td>0.5/γ</td><td>1 × 10-4</td><td>25</td><td>0.05</td><td>{0.25, 1.5, 3}</td><td>{0.84, 0.34, 0.21}</td></tr><tr><td>CIFAR-10(ResNet-18)</td><td>0.5/γ</td><td>1 × 10-4</td><td>100</td><td>0.1</td><td>{0.25, 1.5, 5}</td><td>{1.53, 0.92, 0.40}</td></tr><tr><td>CIFAR-10(VGG-16)</td><td>0.5/γ</td><td>1 × 10-4</td><td>100</td><td>0.1</td><td>{0.25, 1.5, 5}</td><td>{1.23, 0.57, 0.28}</td></tr><tr><td>Tiny ImageNet</td><td>0.5/γ</td><td>2 × 10-5</td><td>100</td><td>0.1</td><td>1.5</td><td>0.93</td></tr></table>

Table 1: Hyperparameters and perturbation ranges on different datasets.

![](images/7a445e3ff785d03c170c884001a81cf1911aff568f830bc27d065187ea02a77b.jpg)

![](images/4744f7fd37479bf32d98e77eb0ffc31177c4a95e1f1218054eeadc360b8ab64e.jpg)

![](images/5bd176eb209d13bcf69931182f9cd5006f46c4fa912e9d3dd8b0cd1670715ac8.jpg)

![](images/3854bc76a7897092bc4b20743317e70fd0fc5915ad36b18655518a790e6ec4c1.jpg)

![](images/d6f12bde1a395ed60006c70b2c24f0f13c6f0519b2cb1387791b80754ca62717.jpg)  
(a)

![](images/7bdd73f76446a8baf82f20d3d9e354419faf8b3829705855d5229514cc158228.jpg)  
(b)

![](images/d9fdbf3ddaa933cf4d1e634ddff0258a913e7e6245f73635406b5050e34af98a.jpg)  
(c)  
Figure 2: (a) gives the distance between the robustness certificate (yellow) and the worst-case performance on testing data (pink) with an example on MNIST. The gap between the two lines indicates the tightness of our certificate (Eq. 3). (b) compares the performance of two models trained with different  $c(\cdot)$ s. The classifier trained with the noise included in the cost has better performance overall. (c) compares the performance of NAL with WRM on MNIST, CNN (ELU) under different  $\gamma$ s. NAL overall has better performance than WRM. (d) compares NAL with SmoothAdv, TRADES and STN on MNIST, CNN at  $\gamma = 0.25$  and the corresponding  $\varepsilon$ . NAL does not show significant improvement when  $\gamma$  is small.

![](images/0920c0e4d8919654a190595153e11dc85b927e97aade6be4e81b7ed881c7a7ed.jpg)  
(d)

and WRM bound the adversarial perturbations by the Wasserstein distance  $\rho$  which is different from the  $\ell_2$ -norm perturbation range  $\varepsilon$  in SmoothAdv and TRADES, we need to establish an equivalence between the perturbation ranges in different methods. Following the convention of Sinha et al. (2018), we choose different  $\gamma$ s and for each  $\gamma$  we generate adversarial examples  $x$  by PGD with 15 iterations. We compute  $\rho$  as the expected transportation cost between the generated adversarial examples and the original inputs over the training set:

$$
\varepsilon^ {2} = \rho (\theta) = \mathbb {E} _ {P _ {0}} \mathbb {E} _ {z} [ c (x + z, x _ {0}) ]. \tag {8}
$$

And  $\varepsilon$  can be computed accordingly. The corresponding values of  $\gamma$  and  $\varepsilon$  used in experiments are given in Table 1 as well.

Attack parameters. To evaluate the empirical accuracies for different methods, we adopt the PGD attack Kurakin et al. (2018); Madry et al. (2017) as the adversarial attack following the convention of Li et al. (2019); Sinha et al. (2018); Zhang et al. (2019), etc. We set the number of iterations in PGD attack as  $K_{\text{attack}} = 20$  and the learning rate  $\eta = 2\varepsilon_{\text{attack}} / K_{\text{attack}}$  where  $\varepsilon_{\text{attack}}$  is  $\ell_2$  attack radius.

# 5.1 RESULTS

Certificate. To better understand how close the upper bound is to the true distributional risk, we plot our certificate  $\gamma \rho + \mathbb{E}_{\widehat{P}_{\mathrm{test}}}[\phi_{\gamma}(\theta; x_0)]$  against any level of robustness  $\rho$ , and the out-of-sample (test) worst-case performance  $\sup_{P \in \mathcal{P}} \mathbb{E}_P[\ell(\theta; \hat{x})]$  for NAL (Fig. 2(a)). Since the worst-case loss is hard to evaluate directly, we solve its Lagrangian relaxation for different values of  $\gamma_{adv}$ . For each  $\gamma_{adv}$ , we compute the average distance to adversarial examples in the test set as  $\widehat{\rho}_{\mathrm{test}}(\theta) := \mathbb{E}_{\widehat{P}_{\mathrm{test}}} \mathbb{E}_z[c(x_\star + z, x_0)]$  where  $\widehat{P}_{\mathrm{test}}$  is the test data distribution and  $x_\star = \mathbb{E}_z \arg \max_x \{\ell(\theta; x + z) - \gamma_{\mathrm{adv}} c(x + z, x_0)\}$  is the adversarial perturbation of  $x_0$ . The worst-case loss is given by  $(\widehat{\rho}_{\mathrm{test}}(\theta), \mathbb{E}_{\widehat{P}_{\mathrm{test}}} \mathbb{E}_z[\ell(\theta; x_\star + z)])$ . As we observe,  $\widehat{\rho}_{\mathrm{test}}(\theta)$  tends to increase with a higher noise level. Hence we need to keep the noise at an appropriate level to make our certificate tractable.

Cost without noise. To find out if NAL works when noise is removed from the cost, we designed a verification experiment on CIFAR-10 (ResNet-18) by letting  $c(x,x_0) = \| x - x_0\| _2^2$  and inserting noise only to  $\ell$ . We set  $\gamma = 1.5$ ,  $\sigma = 0.1$ ,  $K = 4$ ,  $s = 4$ . As Fig. 2(b) has shown, the accuracy performance of the model excluding noise from the cost is far inferior, which shows that the randomized noise is an inherent part in the design.

Sample number and PGD iterations. We also study the impact of the noise sample number  $s$  and PGD iteration  $K$  to the model robustness with CIFAR-10 (ResNet-18) as an example. The result in

![](images/eff767cb5fddd79366189b11085818c445e8aae60e50dba89d324ff663bc1d21.jpg)

![](images/b791bcfd7eb2555e750c307105f8a45f75b2b2dcefc0287dd7bbfc0536c11c37.jpg)

![](images/69798963fad49432e4cc21d6418cee9a81d33419f65062ee273c863c944c0f55.jpg)

![](images/5d46bc18476d52ea00de2e634f98853d188e35ffd01d2277b02f9428aea0dc77.jpg)

![](images/45d164f78a925d040b62c61d8a4b75b33ff6934cce1497ff03045d8c6535e62c.jpg)  
(a)

![](images/341c40542e5e316d9465cc3fd347127d7bbec8384a34918332baf3e9691d3097.jpg)  
(b)

![](images/287c0d4df4ac2a265e0a226eb2639dd2bd85bfddc4a799bb918012eecb98845d.jpg)  
(c)  
Figure 3: NAL outperforms baselines on CIFAR-10, VGG-16 and Tiny ImageNet, ResNet-18. (a),(c) are trained on ELU models under different  $\gamma$ s. For the same  $\gamma$ , NAL exceeds WRM. (b),(d) are trained on ReLU models with  $\gamma = 1.5$  and the corresponding  $\varepsilon$ . NAL yields the highest robustness under different levels of attack. STN has the highest clean accuracy.

![](images/5065cce2b65da21fbd859ed9b86da233e463b2ece7e0ba24347bd0fb773557e7.jpg)  
(d)

Table 2 shows that while the model performance enhances with  $K$ , it does not necessarily increase with a larger noise samples. We did not test with greater noise samples due to high complexity. For a combined consideration of computation overhead and accuracy, we choose  $K = 4$ ,  $s = 4$  by default in the experiments, which is likely to deliver a sufficiently good performance. Due to space constraints, complete experimental results are in Appendix B.2.

Penalty and noise level. We vary the value of  $\gamma$  and  $\sigma$  in the experiments to find out their impact. By the results in Fig. 2 (c), (d) and 3, we observe  $\gamma = 0.25$  yields the best performance for MNIST, and  $\gamma = 1.5$  is best for CIFAR-10 and Tiny ImageNet, considering all levels of adversarial attacks. For a complete result on  $\gamma$ , one can refer to Appendix B.3. Likewise, the best value of  $\sigma$  also depends on the dataset, shown by the experimental results in Appendix B.2.

Comparison with baselines. Finally, we compare the empirical accuracies with the baselines and the results are presented in Fig. 2 (c),(d) and 3. For WRM, the experiments are conducted on the modified structure of DNNs to ensure smoothness. NAL has superior performance in almost all cases except that: 1) the clean accuracies (denoted by  $\ell_2$  attack radius  $= 0$ ) on CIFAR-10 and Tiny ImageNet of NAL are inferior to STN; 2) on MNIST, the performance of NAL is no worse but does not exceed baselines by a large margin. For 1), we found STN mostly has far worse performance than other schemes when the attack radius  $>0$ , which echos the proposition in Salman et al. (2019a) that adversarial training brings higher robustness than stability training. Hence it can be explained by the inherent tradeoff between clean accuracy and robustness (Zhang et al. (2019)) that STN has higher clean accuracies than others. Actually, NAL shows better tradeoff between accuracy and robustness than baselines, indicated by the relatively flat accuracy lines. For 2), we think MNIST has a relatively simple decision boundary than the other two datasets and hence allows larger perturbations (smaller  $\gamma$ ). Thus the performance boost by NAL is not significant. Actually, when  $\gamma$  is larger, the performance of NAL exceeds baselines by a large margin (Appendix B.3).

<table><tr><td>\( \ell_2 \)attack radius</td><td>0</td><td>0.25</td><td>0.5</td><td>0.75</td><td>1</td><td>1.25</td><td>1.5</td><td>1.75</td></tr><tr><td>(K,s)=(4,1)</td><td>0.8647</td><td>0.7540</td><td>0.5950</td><td>0.4297</td><td>0.2814</td><td>0.1713</td><td>0.0983</td><td>0.0520</td></tr><tr><td>(K,s)=(4,4)</td><td>0.8546</td><td>0.7643</td><td>0.6520</td><td>0.5202</td><td>0.3922</td><td>0.2765</td><td>0.1811</td><td>0.1120</td></tr><tr><td>(K,s)=(4,8)</td><td>0.8537</td><td>0.7622</td><td>0.6482</td><td>0.5171</td><td>0.3846</td><td>0.2641</td><td>0.1702</td><td>0.0997</td></tr><tr><td>(K,s)=(8,1)</td><td>0.8593</td><td>0.7555</td><td>0.6091</td><td>0.4630</td><td>0.3260</td><td>0.2205</td><td>0.1453</td><td>0.0929</td></tr><tr><td>(K,s)=(8,4)</td><td>0.8517</td><td>0.7663</td><td>0.6566</td><td>0.5289</td><td>0.3970</td><td>0.2769</td><td>0.1827</td><td>0.1129</td></tr><tr><td>(K,s)=(8,8)</td><td>0.8493</td><td>0.7582</td><td>0.6520</td><td>0.5302</td><td>0.3978</td><td>0.2828</td><td>0.1895</td><td>0.1151</td></tr></table>

Table 2: Testing accuracies of NAL (CIFAR-10, ResNet-18) on a variety of  $s$  and  $K$ . Under each setting, the model with the highest clean accuracy ( $\ell_2$  attack radius = 0) is chosen for testing. Numbers in bold represent the best performance in defending the attack.

# 6 CONCLUSION

Our work view the robustness of a smoothed classifier from a different perspective, i.e., the worst-case population loss over the input distribution. We provide a tractable upper bound (certificate) for the loss and devise a noisy adversarial learning approach to obtain a tight certificate. Compared with previous works, our certificate is practically meaningful and offers superior empirical robustness performance.

# REFERENCES

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. pp. 1310-1320, 2019.  
Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy A Mann, and Pushmeet Kohli. A dual approach to scalable verification of deep networks. In UAI, volume 1, pp. 2, 2018.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. 2015.  
Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. arXiv preprint arXiv:1810.12715, 2018.  
Zhezhi He, Adnan Siraj Rakin, and Deliang Fan. Parametric noise injection: Trainable randomness to improve deep neural network robustness against adversarial attack. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 588-597, 2019.  
Jinyuan Jia, Xiaoyu Cao, Binghui Wang, and Neil Zhenqiang Gong. Certified robustness for top-k predictions against adversarial perturbations via randomized smoothing. In International Conference on Learning Representations, 2019.  
Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adversarial logit pairing. arXiv preprint arXiv:1803.06373, 2018.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale. 2018.  
Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified robustness to adversarial examples with differential privacy. In 2019 IEEE Symposium on Security and Privacy (SP), pp. 656-672. IEEE, 2019.  
Guang-He Lee, Yang Yuan, Shiyu Chang, and Tommi Jaakkola. Tight certificates of adversarial robustness for randomly smoothed classifiers. In Advances in Neural Information Processing Systems, pp. 4910-4921, 2019.  
Bai Li, Changyou Chen, Wenlin Wang, and Lawrence Carin. Certified adversarial robustness with additive noise. In Advances in Neural Information Processing Systems, pp. 9464-9474, 2019.  
Xuanqing Liu, Minhao Cheng, Huan Zhang, and Cho-Jui Hsieh. Towards robust neural networks via random self-ensemble. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 369–385, 2018.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. 2017.  
Matthew Mirman, Timon Gehr, and Martin Vechev. Differentiable abstract interpretation for provably robust neural networks. In International Conference on Machine Learning, pp. 3578-3586, 2018.  
Hongseok Namkoong and John C Duchi. Variance-based regularization with convex objectives. In Advances in neural information processing systems, pp. 2971-2980, 2017.  
NhatHai Phan, My T Thai, Han Hu, Ruoming Jin, Tong Sun, and Dejing Dou. Scalable differential privacy with certified robustness in adversarial learning. 2020.  
Rafael Pinot, Laurent Meunier, Alexandre Araujo, Hisashi Kashima, Florian Yger, Cédric Gouy-Pailler, and Jamal Atif. Theoretical evidence for adversarial robustness through randomization. In Advances in Neural Information Processing Systems, pp. 11838-11848, 2019.  
Rafael Pinot, Raphael Ettedgui, Geovani Rizk, Yann Chevaleyre, and Jamal Atif. Randomization matters. how to defend against strong adversarial attacks. 2020.

Aditi Raghunathan, Jacob Steinhardt, and Percy S Liang. Semidefinite relaxations for certifying robustness to adversarial examples. In Advances in Neural Information Processing Systems, pp. 10877-10887, 2018.  
Hadi Salman, Jerry Li, Ilya Razenshteyn, Pengchuan Zhang, Huan Zhang, Sebastien Bubeck, and Greg Yang. Provably robust deep learning via adversarially trained smoothed classifiers. In Advances in Neural Information Processing Systems, pp. 11292-11303, 2019a.  
Hadi Salman, Greg Yang, Huan Zhang, Cho-Jui Hsieh, and Pengchuan Zhang. A convex relaxation barrier to tight robustness verification of neural networks. In Advances in Neural Information Processing Systems, pp. 9835-9846, 2019b.  
Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Puschel, and Martin Vechev. Fast and effective robustness certification. In Advances in Neural Information Processing Systems, pp. 10802-10813, 2018.  
Aman Sinha, Hongseok Namkoong, and John Duchi. Certifying some distributional robustness with principled adversarial training. 2018.  
Vincent Tjeng, Kai Y Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed integer programming. In International Conference on Learning Representations, 2018.  
Yisen Wang, Difan Zou, Jinfeng Yi, James Bailey, Xingjun Ma, and Quanquan Gu. Improving adversarial robustness requires revisiting misclassified examples. In International Conference on Learning Representations, 2019.  
Tsui-Wei Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Luca Daniel, and Inderjit Dhillon. Towards fast computation of certified robustness for relu networks. In International Conference on Machine Learning (ICML), 2018.  
Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In International Conference on Machine Learning, pp. 5286-5295. PMLR, 2018.  
Valentina Zantedeschi, Maria-Irina Nicolae, and Ambrish Rawat. Efficient defenses against adversarial attacks. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, pp. 39-49, 2017.  
Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael I Jordan. Theoretically principled trade-off between robustness and accuracy. 2019.  
Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. In Advances in neural information processing systems, pp. 4939-4948, 2018.  
Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li, Duane Boning, and Cho-Jui Hsieh. Towards stable and efficient training of verifiably robust neural networks. 2020.  
Stephan Zheng, Yang Song, Thomas Leung, and Ian Goodfellow. Improving the robustness of deep neural networks via stability training. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4480-4488, 2016.
