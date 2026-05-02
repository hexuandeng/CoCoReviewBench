# REVISITING DEEPFOOL: GENERALIZATION AND IMPROVEMENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal  $\ell_2$  adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal  $\ell_2$  adversarial perturbations.

# 1 INTRODUCTION

Deep learning has achieved breakthrough improvement in numerous tasks and has developed as a powerful tool in various applications, including computer vision Long et al. (2015) and speech processing Mikolov et al. (2011). Despite their success, deep neural networks are known to be vulnerable to adversarial examples, carefully perturbed examples perceptually indistinguishable from original samples Szegedy et al. (2013). This can lead to a significant disruption of the inference result of deep neural networks. It has important implications for safety and security-critical applications of machine learning models.

Our goal in this paper is to introduce a parameter-free and simple method for accurately and reliably evaluating the adversarial robustness of deep networks in a fast and geometrically-based fashion. Most of the current attack methods rely on general-purpose optimization techniques, such as Projected Gradient Descent (PGD) Madry et al. (2017) and Augmented Lagrangian Rony et al. (2021), which are oblivious to the geometric properties of models. However, deep neural networks' robustness

to adversarial perturbations is closely tied to their geometric landscape Dauphin et al. (2014); Poole et al. (2016). Given this, it would be beneficial to exploit such properties when designing and implementing adversarial attacks. This allows to create more effective and computationally efficient attacks on classifiers. Formally, for a given classifier  $\hat{k}$  and input  $\pmb{x}$ , we define an adversarial perturbation as

![](images/3d2516d694afef577f83a532c69b7d74b2bdc6ca7de2b3ce719338d97c1de671.jpg)  
Figure 1: The average number of gradient computations vs the mean  $\ell_2$ -norm of perturbations. It shows that our novel fast and accurate method, SDF, outperforms other minimum-norm attacks. SDF finds significantly smaller perturbations compared to DF, with only a small increase in computational cost. SDF also outperforms other algorithms in optimality and speed. The numbers are taken from Table 5.

the minimal perturbation  $\pmb{r}$  that is sufficient to change the estimated label  $\hat{k}(\pmb{x})$ :

$$
\Delta (\boldsymbol {x}; \hat {k}) := \min  _ {\boldsymbol {r}} \| \boldsymbol {r} \| _ {2} \text {s . t} \hat {k} (\boldsymbol {x} + \boldsymbol {r}) \neq \hat {k} (\boldsymbol {x}). \tag {1}
$$

DeepFool (DF) Moosavi-Dezfooli et al. (2016) was among the earliest attempts to exploit the "excessive linearity" Goodfellow et al. (2014) of deep networks to find minimum-norm adversarial perturbations. However, more sophisticated attacks were later developed that could find smaller perturbations at the expense of significantly greater computation time.

In this paper, we exploit the geometric characteristics of minimum-norm adversarial perturbations to design a family of fast yet simple algorithms that achieves a better trade-off between computational cost and accuracy in finding  $\ell_2$  adversarial perturbations (see Fig. 1). Our proposed algorithm, guided by the characteristics of the optimal solution to Eq. (1), enhances DF to obtain smaller perturbations, while maintaining simplicity and computational efficiency that are only slightly inferior to those of DF. Our main contributions are summarized as follows:

- We introduce a novel family of fast yet accurate algorithms to find minimal adversarial perturbations. We extensively evaluate and compare our algorithms with state-of-the-art attacks in various settings.  
- Our algorithms are developed in a systematic and well-grounded manner, based on theoretical analysis.  
- We further improve the robustness of state-of-the-art image classifiers to minimum-norm adversarial attacks via adversarial training on the examples obtained by our algorithms.  
- We significantly improve the time efficiency of the state-of-the-art Auto-Attack (AA) Croce & Hein (2020b) by adding our proposed method to the set of attacks in AA.

Related works. It has been observed that deep neural networks are vulnerable to adversarial examples Szegedy et al. (2013); Moosavi-Dezfooli et al. (2016); Goodfellow et al. (2014). To exploit this vulnerability, a range of methods have been developed for generating adversarial perturbations for image classifiers. These attacks occur in two settings: white-box, where the attacker has complete knowledge of the model, including its architecture, parameters, defense mechanisms, etc.; and black-box, where the attacker's knowledge is limited, mostly relying on input queries to observe outputs Chen et al. (2020); Rahmati et al. (2020). Further, adversarial attacks can be broadly categorized into two categories: bounded-norm attacks (such as FGSM Goodfellow et al. (2014) and PGD Madry et al. (2017)) and minimum-norm attacks (such as DF and C&W Carlini & Wagner (2017)) with the latter aimed at solving Eq. (1). In this work, we specifically focus on white-box minimum  $\ell_2$ -norm attacks.

The authors in Szegedy et al. (2013) studied adversarial examples by solving a penalized optimization problem. The optimization approach used in Szegedy et al. (2013) is complex and computationally inefficient; therefore, it cannot scale to large datasets. The method proposed in Goodfellow et al. (2014) applied a single-step of the input gradient to generate adversarial examples efficiently. DF was the first method to seek minimum-norm adversarial perturbations, employing an iterative approach. It linearizes the classifier at each step to estimate the minimal adversarial perturbations efficiently. C&W attack Carlini & Wagner (2017) transform the optimization problem in Szegedy et al. (2013) into an unconstrained optimization problem. C&W leverages the first-order gradient-based optimizers to minimize a balanced loss between the norm of the perturbation and misclassification confidence. Inspired by the geometric idea of DF, FAB Croce & Hein (2020a) presents an approach to minimize the norm of adversarial perturbations by employing complex projections and approximations while maintaining proximity to the decision boundary. By utilizing gradients to estimate the local geometry of the boundary, this method formulates minimum-norm optimization without the need for tuning a weighting term. DDN Rony et al. (2019) uses projections on the  $\ell_2$ -ball for a given perturbation budget  $\epsilon$ . FMN Pintor et al. (2021) extends the DDN attack to other  $\ell_p$ -norms. By formulating (1) with Lagrange's method, ALMA Rony et al. (2021) introduced a framework for finding adversarial examples for several distances.

# 2 DEEPFOOL (DF) AND MINIMAL ADVERSARIAL PERTURBATIONS

In this section, we first discuss the geometric interpretation of the minimum-norm adversarial perturbations, i.e., solutions to the optimization problem in Eq. (1). We then examine DF to demonstrate

why it may fail to find the optimal minimum-norm perturbation. Then in the next section, we introduce our proposed method that exploits DF to find smaller perturbations.

Let  $f: \mathbb{R}^d \to \mathbb{R}^C$  denote a  $C$ -class classifier, where  $f_k$  represents the classifier's output associated to the  $k$ th class. Specifically, for a given datapoint  $\pmb{x} \in \mathbb{R}^d$ , the estimated label is obtained by  $\hat{k}(\pmb{x}) = \operatorname{argmax}_k f_k(\pmb{x})$ , where  $f_k(\pmb{x})$  is the  $k^{\text{th}}$  component of  $f(\pmb{x})$  that corresponds to the  $k^{\text{th}}$  class. Note that the classifier  $f$  can be seen as a mapping that partitions the input space  $\mathbb{R}^d$  into classification regions, each of which has a constant estimated label (i.e.,  $\hat{k}(.)$  is constant for each such region). The decision boundary  $\mathcal{B}$  is defined as the set of points in  $\mathbb{R}^d$  such that  $f_i(\pmb{x}) = f_j(\pmb{x}) = \max_k f_k(\pmb{x})$  for some distinct  $i$  and  $j$ .

Additive  $\ell_2$ -norm adversarial perturbations are inherently related to the geometry of the decision boundary. More formally, Let  $\boldsymbol{x} \in \mathbb{R}^d$ , and  $\boldsymbol{r}^{*}(\boldsymbol{x})$  be the minimal adversarial perturbation defined as the minimizer of Eq. (1). Then  $\boldsymbol{r}^{*}(\boldsymbol{x})$ , 1) is orthogonal to the decision boundary of the classifier  $\mathcal{B}$ , and 2) its norm  $\| \boldsymbol{r}^{*}(\boldsymbol{x}) \|_2$  measures the Euclidean distance between  $\boldsymbol{x}$  and  $\mathcal{B}$ , that is  $\boldsymbol{x} + \boldsymbol{r}^{*}$  lies on  $\mathcal{B}$ . We aim to investigate whether the perturbations generated by DF satisfy the aforementioned two conditions. Let  $\boldsymbol{r}_{\mathrm{DF}}$  denote the perturbation found by DF for a datapoint  $\boldsymbol{x}$ . We expect  $\boldsymbol{x} + \boldsymbol{r}_{\mathrm{DF}}$  to lie on the decision boundary. Hence, if  $\boldsymbol{r}$  is the minimal perturbation, for all  $0 < \gamma < 1$ , we expect the perturbation  $\gamma \boldsymbol{r}$  to remain in the same decision region as of  $\boldsymbol{x}$  and thus fail to fool the model.

In Fig. 2 (top-left), we consider the fooling rate of  $\gamma r_{\mathrm{DF}}$  for  $0.2 < \gamma < 1$ . For a minimum-norm perturbation, we expect an immediate sharp decline for  $\gamma$  close to one. However, in Fig. 2 (top-left) we cannot observe such a decline (a sharp decline happens close to  $\gamma = 0.9$ , not 1). This is a confirmation that DF typically finds an overly perturbed point. One potential reason for this is the fact that DF stops when a misclassified point is found, and this point might be an overly perturbed one within the adversarial region, and not necessarily on the decision boundary.

Now, let us consider the other characteristic of the minimal adversarial perturbation. That is, the perturbation should be orthogonal to the decision boundary. We measure the angle between the found perturbation  $r_{\mathrm{DF}}$  and the normal vector orthogonal to the decision boundary  $(\nabla f(\pmb{x} + \pmb{r}_{\mathrm{DF}}))$ . To do so, we first scale  $r_{\mathrm{DF}}$  such that  $\pmb{x} + \gamma r_{\mathrm{DF}}$  lies on the decision boundary. It can be simply done via performing a line search along  $r_{\mathrm{DF}}$ . We then compute the cosine of the angle between  $r_{\mathrm{DF}}$  and the normal to the decision boundary at  $\pmb{x} + \gamma r_{\mathrm{DF}}$  (this angle is denoted by  $\cos(\alpha)$ ). A necessary condition for  $\gamma r_{\mathrm{DF}}$  to be an optimal perturbation is that it must be parallel to the normal vector of the decision boundary (Figure 6 in the Appendix). In Fig. 2 (top-right), we show the distribution of cosine of this angle. Ideally, we wanted this distribution to be accumulated around one. However, it clearly shows that this is not the case, which is a confirmation that  $r_{\mathrm{DF}}$  is not necessarily the minimal perturbation.

# 3 SUPERDEEPFOOL: EFFICIENT ALGORITHMS TO FIND MINIMAL PERTURBATIONS

In this section, we propose a new class of methods that modifies DF to address the aforementioned challenges in the previous section. The goal is to maintain the desired characteristics of DF, i.e., computational efficiency and the fact that it is parameter-free while finding smaller adversarial perturbations. We achieve this by introducing an additional projection step which its goal is to steer the direction of perturbation towards the optimal solution of Eq. (1).

Let us first briefly recall how DF finds an adversarial perturbations for a classifier  $f$ . Given the current point  $x_{i}$ , DF updates it according to the following equation:

$$
\boldsymbol {x} _ {i + 1} = \boldsymbol {x} _ {i} - \frac {f (\boldsymbol {x} _ {i})}{\| \nabla f (\boldsymbol {x} _ {i}) \| _ {2} ^ {2}} \nabla f (\boldsymbol {x} _ {i}). \tag {2}
$$

Here the gradient is taken w.r.t. the input. The intuition is that, in each iteration, DF finds the minimum perturbation for a linear classifier that approximates the model around  $x_{i}$ . The below proposition shows that under certain conditions, repeating this update step eventually converges to a point on the decision boundary.

Proposition 1 Let the binary classifier  $f: \mathbb{R}^d \to \mathbb{R}$  be continuously differentiable and its gradient  $\nabla f$  and  $f$  are  $L'$ -Lipschitz. For a given input sample  $\pmb{x}_0$ , suppose  $B(\pmb{x}_0, \epsilon)$  is a ball centered around

![](images/f84c9692f6f4a09f1c7b18c37311d5505620a37d91c12deb276999d27816681a.jpg)

![](images/cd36469f18d26ca6e182716259660b2874c2919ec5608a969f491c300654834f.jpg)

![](images/5106beb38a981f28eabc831a0db301af5ad3dc5ba1dcb938dd9fbcbfd37f7d33.jpg)  
Figure 2: (Left) we generated 1000 images with one hundred  $\gamma$  between zero and one, and the fooling rate of the DeepFool (top) and SuperDeepFool (bottom) is reported. This experiment is done on the CIFAR10 dataset and ResNet18 model. (Right) histogram of the cosine angle between the normal to the decision boundary and the perturbation vector obtained by DeepFool (top) and SuperDeepFool (bottom) has been showed.

![](images/7319ac0c4f4d9380fbea95d90b04701afdf7848635870707ada822b039647be3.jpg)

$\pmb{x}_0$  with radius  $\epsilon$ , such that there exists  $\pmb{x} \in B(\pmb{x}_0, \epsilon)$  that  $f(\pmb{x}) = 0$ . If  $\|\nabla f\|_2 \geq \zeta$  for all  $\pmb{x} \in B$  and  $\epsilon < \frac{\zeta^2}{L'^2}$ , then DF iterations converge to a point on the decision boundary.

Proof: We defer the proof to the Appendix.

Notice while the proposition guarantees the perturbed sample to lie on the decision boundary, it does not state anything about the orthogonality of the perturbation to the decision boundary.

To find perturbations that are more aligned with the normal to the decision boundary, we introduce an additional projection step that steers the perturbation direction towards the optimal solution of Eq. (1). Formally, the optimal perturbation,  $\boldsymbol{r}^*$ , and the normal to the decision boundary at  $\boldsymbol{x}_0 + \boldsymbol{r}^*$ ,  $\nabla f(\boldsymbol{x}_0 + \boldsymbol{r}^*)$ , should be parallel. Equivalently,  $\boldsymbol{r}^*$  should be a solution of the following maximization problem:

$$
\max  _ {\boldsymbol {r}} \frac {\boldsymbol {r} ^ {\top} \nabla f (\boldsymbol {x} _ {0} + \boldsymbol {r})}{\| \nabla f (\boldsymbol {x} _ {0} + \boldsymbol {r}) \| \| \boldsymbol {r} \|}, \tag {3}
$$

which is the cosine of the angle between  $\mathbf{r}$  and  $\nabla f(\mathbf{x}_0 + \mathbf{r})$ . A necessary condition for  $\mathbf{r}^*$  to be a solution of Eq. (3) is that the projection of  $\mathbf{r}^*$  on the subspace orthogonal to  $\nabla f(\mathbf{x}_0 + \mathbf{r}^*)$  should be zero. Then,  $\mathbf{r}^*$  can be seen as a fixed point of the following iterative map:

$$
\boldsymbol {r} _ {i + 1} = T (\boldsymbol {r} _ {i}) = \frac {\boldsymbol {r} _ {i} ^ {\top} \nabla f \left(\boldsymbol {x} _ {0} + \boldsymbol {r} _ {i}\right)}{\| \nabla f \left(\boldsymbol {x} _ {0} + \boldsymbol {r} _ {i}\right) \|} \cdot \frac {\nabla f \left(\boldsymbol {x} _ {0} + \boldsymbol {r} _ {i}\right)}{\| \nabla f \left(\boldsymbol {x} _ {0} + \boldsymbol {r} _ {i}\right) \|}. \tag {4}
$$

The scalar multiplier on the right-hand side of Eq. (4) represents the norm of the projection of the vector  $\mathbf{r}_i$  along the gradient direction. The following proposition shows that this iterative process can converge to a solution of Eq. (3).

Proposition 2 For a differentiable  $f$  and a given  $\mathbf{r}_0$ ,  $\mathbf{r}_i$  in the iterations Eq. (4) either converge to a solution of Eq. (3) or a trivial solution (i.e.,  $\mathbf{r}_i \to 0$ ).

Proof: We defer the proof to the Appendix.

# 3.1 A FAMILY OF ADVERSARIAL ATTACKS

Algorithm 1: SDF  $(m,n)$  for binary classifiers

Input: image  $x_0$ , classifier  $f, m$ , and  $n$ .

Output: perturbation  $r$

1 Initialize:  $\pmb{x} \gets \pmb{x}_0$  
2 while  $\operatorname{sign}(f(\pmb{x})) = \operatorname{sign}(f(\pmb{x}_0))$  do  
3 repeat  $m$  times  
4  $\pmb {x}\gets \pmb {x} - \frac{|f(\pmb{x})|}{\|\nabla f(\pmb {x})\|_2^2}\nabla f(\pmb {x})$  
5 end  
6 repeat  $n$  times  
7  $\pmb{x} \gets \pmb{x}_0 + \frac{(\pmb{x} - \pmb{x}_0)^\top \nabla f(\pmb{x})}{\|\nabla f(\pmb{x})\|^2} \nabla f(\pmb{x})$  
8 end  
9 end  
10 return  $\pmb {r} = \pmb {x} - \pmb{x}_0$

Finding minimum-norm adversarial perturbations can be seen as a multi-objective optimization problem, where we want  $f(\boldsymbol{x} + \boldsymbol{r}) = 0$  and the perturbation  $\boldsymbol{r}$  to be orthogonal to the decision boundary. So far we have seen that DF finds a solution satisfying the former objective and the iterative map Eq. (4) can be used to find a solution for the latter. A natural approach to satisfy both objectives is to alternate between these two iterative steps, namely Eq. (2) and Eq. (4). We propose a family of adversarial attack algorithms, coined SuperDeepFool, by varying how frequently we alternate between these two steps. We denote this family of algorithms with SDF(m,n), where  $m$  is the number of DF steps Eq. (2) followed by  $n$  repetition of the projection step Eq. (4). This process is summarized in Algorithm 1. One interesting case is SDF(∞,1) which, in each iteration, continues DF steps till a point on the decision boundary is found and then applies the projection step. This

particular case has a resemblance with the strategy used in Rahmati et al. (2020) to find black-box adversarial perturbations. This algorithm can be interpreted as iteratively approximating the decision boundary with a hyperplane and then analytically calculating the minimal adversarial perturbation for a linear classifier for which this hyperplane is the decision boundary. It is justified by the observation that the decision boundary of state-of-the-art deep networks has a small mean curvature around data samples Fawzi et al. (2017; 2018). A geometric illustration of this procedure is shown in Figure 3.

# 3.2 SDF ATTACK

We empirically compare the performance of  $\mathrm{SDF}(m,n)$  for different values of  $m$  and  $n$  in Section 4.1. Interestingly, we observe that we get better attack performance when we apply several DF steps followed by a single projection. Since the standard DF typically finds an adversarial example in less than four iterations for state-of-the-art image classifiers, one possibility is to continue DF steps till an adversarial example is found and then apply a single projection step. We simply call this particular version  $\mathrm{SDF}(\infty,1)$  of our algorithm SDF, which we will extensively evaluate in Section 7. SDF can be understood as a generic algorithm that can also work for the multi-class case by simply substituting the first inner loop of Algorithm 1 with the standard multi-class DF algorithm. The label of the obtained adversarial example determines the boundary on which the projection step will be performed. A summary of multi-class SDF is presented in Algorithm 2. Compared to the standard DF, this algorithm has an additional projection step. We will see later that such a simple modification leads to significantly smaller perturbations.

Algorithm 2: SDF for multi-class classifiers

Input: image  $x_0$ , classifier  $f$ .

Output: perturbation  $r$

1 Initialize:  $x\gets x_0$  
2 while  $\hat{k} (\pmb {x}) = \hat{k} (\pmb {x}_0)$  do  
3  $\widetilde{\pmb{x}}\gets$  DeepFool(x)  
4  $\pmb{w} \gets \nabla f_{\hat{k}(\widetilde{\pmb{x}})}(\widetilde{\pmb{x}}) - \nabla f_{\hat{k}(\pmb{x}_0)}(\widetilde{\pmb{x}})$  
5  $\pmb{x} \gets \pmb{x}_0 + \frac{(\widetilde{\pmb{x}} - \pmb{x}_0)^\top \pmb{w}}{\|\pmb{w}\|^2} \pmb{w}$  
6 end  
7 return  $r = x - x_0$

![](images/e398e02fd0586c584c52ec1dc8fdb1972147276c5cfde76c8adf4a4958a897f6.jpg)  
Figure 3: Illustration of two iterations of the SDF  $(\infty,1)$  algorithm. Here  $\mathbf{x}_0$  is the original data point and  $\mathbf{x}_{*}$  is the minimum-norm adversarial example.

Table 1: The cosine of the angle between the perturbation vector  $(\boldsymbol{r})$  and  $\nabla f(\boldsymbol{x} + \boldsymbol{r})$ . We performed this experiment on three models trained on CIFAR10 dataset.  

<table><tr><td rowspan="2">Attack</td><td colspan="3">Models</td></tr><tr><td>LeNet</td><td>ResNet18</td><td>WRN-28-10</td></tr><tr><td>DF</td><td>0.89</td><td>0.14</td><td>0.21</td></tr><tr><td>SDF (1,1)</td><td>0.90</td><td>0.63</td><td>0.64</td></tr><tr><td>SDF (1,3)</td><td>0.88</td><td>0.61</td><td>0.62</td></tr><tr><td>SDF (3,1)</td><td>0.92</td><td>0.70</td><td>0.72</td></tr><tr><td>SDF (∞,1)</td><td>0.92</td><td>0.72</td><td>0.80</td></tr></table>

Table 2: Comparison of the  $\ell_2$ -norm of perturbations for DF and SDF family algorithms. We performed this experiment on CIFAR10. We use the same model architecture and hyperparameters for training as in C&W and DDN.  

<table><tr><td>Attack</td><td>Mean-ℓ2</td><td>Median-ℓ2</td><td>Grads</td></tr><tr><td>DF</td><td>0.17</td><td>0.15</td><td>14</td></tr><tr><td>SDF (1,1)</td><td>0.14</td><td>0.13</td><td>22</td></tr><tr><td>SDF (1,3)</td><td>0.16</td><td>0.14</td><td>26</td></tr><tr><td>SDF (3,1)</td><td>0.12</td><td>0.11</td><td>30</td></tr><tr><td>SDF(∞,1)</td><td>0.11</td><td>0.10</td><td>32</td></tr></table>

# 4 EXPERIMENTAL RESULTS

In this section, we conduct extensive experiments to demonstrate the effectiveness of our method in different setups and for several natural and adversarially trained networks. We first introduce our experimental settings, including datasets, models, and attacks. Next, we compare our method with state-of-the-art  $\ell_2$ -norm adversarial attacks in various settings, demonstrating the superiority of our simple yet fast algorithm for finding accurate adversarial examples. Moreover, we add SDF to the collection of attacks used in AutoAttack, and call the new set of attacks Auto-Attack++. This setup meaningfully speeds up the process of finding norm-bounded adversarial perturbations. We also demonstrate that a model adversarially training using the SDF perturbations becomes more robust compared to the models<sup>1</sup> trained using other minimum-norm attacks.

Setup. We test our algorithms on architectures trained on MNIST, CIFAR10, and ImageNet datasets. For MNIST, we use a robust model called IBP from Zhang et al. (2019) and naturally trained model called SmallCNN. For CIFAR10, we use three models: an adversarially trained PreActResNet-18 He et al. (2016b) from Rade & Moosavi-Dezfooli (2021), a regularly trained Wide ResNet 28-10 (WRN-28-10) from Zagoruyko & Komodakis (2016) and LeNet LeCun et al. (1999). These models are obtainable via the RobustBench library Croce et al. (2020). On ImageNet, we test the attacks on two ResNet-50 (RN-50) models: one regularly trained and one  $\ell_2$  adversarially trained, obtainable through the robustness library Engstrom et al. (2019).

# 4.1 COMPARISON WITH DEEPFOOL (DF)

In this part, we compare our algorithm in terms of orthogonality and size of the  $\ell_2$ -norm perturbations especially with DF. Assume  $\boldsymbol{r}$  is the perturbation vector obtained by an adversarial attack. First, we measure the orthogonality of perturbations by measuring the inner product between  $\nabla f(\boldsymbol{x} + \boldsymbol{r})$  and  $\boldsymbol{r}$ . As we explained in Section 2, a larger inner product between  $\boldsymbol{r}$  and the gradient vector at  $f(\boldsymbol{x} + \boldsymbol{r})$  indicates that the perturbation vector is closer to the optimal perturbation vector  $\boldsymbol{r}^*$ . We compare the orthogonality of different members of the SDF family and DF. The results are shown in Table 1. We observe that DF finds perturbations orthogonal to the decision boundary for low-complexity models such as LeNet, but fails to perform effectively when evaluated against more complex ones. In contrast, attacks from the SDF family consistently found perturbations with a larger cosine of the angle for all three models.

Verifying optimality conditions for SDF. We validate the optimality conditions of the perturbations generated by SDF using the procedure outlined in Section 2. Comparing Fig. 2 top (DF) and bottom (SDF) rows, it becomes evident that our approach effectively mitigates the two issues we previously highlighted for DF. Namely, the alignment of the perturbation with the normal to the decision boundary and the problem of over-perturbation. We can see that unlike DF, the cosine of the angle for SDF is more concentrated around one, which indicates that the SDF perturbations are more aligned with the normal to the decision boundary. Moreover, Fig. 2 (bottom-right) shows a sharper decline in the fooling rate (going down quickly to zero) when  $\gamma$  decreases. This is consistent with

Table 3: Performance for attacks on the MNIST dataset with IBP models. The numbers between parentheses indicate the number of iterations.  

<table><tr><td>Attack</td><td>FR</td><td>Median-ℓ2</td><td>Grads</td></tr><tr><td>ALMA (1000)</td><td>100</td><td>1.26</td><td>1000</td></tr><tr><td>ALMA (100)</td><td>98.90</td><td>4.96</td><td>100</td></tr><tr><td>DDN (1000)</td><td>99.27</td><td>1.46</td><td>1000</td></tr><tr><td>DDN (100)</td><td>94.34</td><td>1.97</td><td>100</td></tr><tr><td>FAB (1000)</td><td>99.98</td><td>3.34</td><td>10000</td></tr><tr><td>FAB (100)</td><td>99.98</td><td>5.19</td><td>1000</td></tr><tr><td>FMN (1000)</td><td>89.08</td><td>1.34</td><td>1000</td></tr><tr><td>FMN (100)</td><td>67.80</td><td>2.14</td><td>100</td></tr><tr><td>C&amp;W</td><td>4.63</td><td>-</td><td>90000</td></tr><tr><td>SDF</td><td>100</td><td>1.37</td><td>52</td></tr></table>

Table 4: Performance of attacks on the CIFAR-10 dataset with WRN-28-10. The results on adversarially trained networks are deferred to Tables 20 and 11 of the Appendix.  

<table><tr><td>Attacks</td><td>FR</td><td>Median-ℓ2</td><td>Grads</td></tr><tr><td>DF</td><td>100</td><td>0.26</td><td>14</td></tr><tr><td>ALMA</td><td>100</td><td>0.10</td><td>100</td></tr><tr><td>DDN</td><td>100</td><td>0.13</td><td>100</td></tr><tr><td>FAB</td><td>100</td><td>0.11</td><td>100</td></tr><tr><td>FMN</td><td>97.3</td><td>0.11</td><td>100</td></tr><tr><td>C&amp;W</td><td>100</td><td>0.12</td><td>90000</td></tr><tr><td>SDF</td><td>100</td><td>0.09</td><td>25</td></tr></table>

our expectation for an accurate minimal perturbation attack. Table 2 demonstrates that SDF family outperforms DF in finding more accurate perturbations, particularly  $\mathrm{SDF}(\infty ,1)$  which significantly outperforms DF at a small cost.

# 4.2 COMPARISON WITH MINIMUM-NORM ATTACKS

We now compare SDF with state-of-the-art minimum  $\ell_2$ -norm attacks: C&W, FMN, DDN, ALMA, and FAB. For C&W, we use the same hyperparameters as in Rony et al. (2019). We use FMN, FAB, DDN, and ALMA with budgets of 100 and 1000 iterations and report the best performance. For a fair comparison, we clip the pixel-values of SDF-generated adversarial images to [0, 1], consistent with the other minimum-norm attacks. We report the average number of gradient computations per sample, as these operations are computationally intensive and provide a consistent metric unaffected by hardware differences. We also provide a runtime comparison in Table 20 of the Appendix.

We evaluate the robustness of the IBP model, which is adversarially trained on the MNIST dataset, against state-of-the-art attacks in Table 3. We choose this robust model as it allows us to have a more nuanced comparison between different adversarial attacks. SDF and ALMA are the only attacks that achieve a  $100\%$  percent fooling rate against this model, whereas C&W is unsuccessful on most of the data samples. The fooling rates of the remaining attacks also degrade when evaluated with 100 iterations. For instance, FMN's fooling rate decreases from  $89\%$  to  $67.8\%$  when the number of iterations is reduced from 1000 to 100. This observation shows that, unlike SDF, selecting the necessary number of iterations is critical for the success of fixed-iteration attacks. Even for ALMA which can achieve a nearly perfect FR, decreasing the number of iterations from 1000 to 100 causes the median norm of perturbations to increase fourfold. In contrast, SDF is able to compute adversarial perturbations using the fewest number of gradient computations while still outperforming the other algorithms, except ALMA, in terms of the perturbation norm. However, it is worth noting that ALMA requires twenty times more gradient computations compared to SDF to achieve a marginal improvement in the perturbation norm.

Table 4 compares SDF with state-of-the-art attacks on the CIFAR10 dataset. The results show that state-of-the-art attacks have a similar norm of perturbations, but an essential point is the speed of attacks. SDF finds more accurate adversarial perturbation very quickly rather than other algorithms. We also evaluated all attacks on an adversarially trained model for the CIFAR10 dataset. SDF achieves smaller perturbations with half the gradient calculations than other attacks. SDF finds smaller adversarial perturbations for adversarially trained networks at a significantly lower cost than other attacks, requiring only  $20\%$  of FAB's cost and  $50\%$  of DDN's and ALMA's (see Tables 11 and 20 in the Appendix).

Table 5 demonstrates the performance of SDF on a naturally and adversarially trained models on ImageNet dataset. Unlike models trained on CIFAR10, where the attacks typically result in perturbations with similar norm, the differences between attacks are more nuanced for ImageNet

Table 5: Performance comparison of SDF with other state-of-the-art attacks for median  $\ell_2$  on ImageNet dataset. FR columns show the fooling rates of attacks.  

<table><tr><td rowspan="2">Attack</td><td colspan="3">RN-50</td><td colspan="3">RN-50 (AT)</td></tr><tr><td>FR</td><td>Median-ℓ2</td><td>Grads</td><td>FR</td><td>Median-ℓ2</td><td>Grads</td></tr><tr><td>DF</td><td>99.1</td><td>0.31</td><td>23</td><td>98.8</td><td>1.36</td><td>34</td></tr><tr><td>ALMA</td><td>100</td><td>0.10</td><td>100</td><td>100</td><td>0.85</td><td>100</td></tr><tr><td>DDN</td><td>99.9</td><td>0.17</td><td>1,000</td><td>99.7</td><td>1.10</td><td>1,000</td></tr><tr><td>FAB</td><td>99.3</td><td>0.10</td><td>900</td><td>100</td><td>0.81</td><td>900</td></tr><tr><td>FMN</td><td>99.3</td><td>0.10</td><td>1,000</td><td>99.9</td><td>0.82</td><td>1,000</td></tr><tr><td>C&amp;W</td><td>100</td><td>0.21</td><td>82,667</td><td>99.9</td><td>1.17</td><td>52,000</td></tr><tr><td>SDF</td><td>100</td><td>0.09</td><td>37</td><td>100</td><td>0.80</td><td>49</td></tr></table>

Table 6: The comparison between  $\ell_2$  robustness of our adversarial trained model and Rony et al. (2019) model. We perform this experiment on CIFAR10 dataset.  

<table><tr><td rowspan="2">Attack</td><td colspan="2">SDF (Ours)</td><td colspan="2">DDN</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>DDN</td><td>1.09</td><td>1.02</td><td>0.86</td><td>0.73</td></tr><tr><td>FAB</td><td>1.12</td><td>1.03</td><td>0.92</td><td>0.75</td></tr><tr><td>FMN</td><td>1.48</td><td>1.43</td><td>1.47</td><td>1.43</td></tr><tr><td>ALMA</td><td>1.17</td><td>1.06</td><td>0.84</td><td>0.71</td></tr><tr><td>SDF</td><td>1.06</td><td>1.01</td><td>0.81</td><td>0.73</td></tr></table>

Table 7: Average input curvature of WRN-28-10 models trained on CIFAR10 dataset, according to the measures proposed in Srinivas et al.. The second column shows the average spectralnorm of the Hessian w.r.t. input,  $\| \nabla^2 f(\mathbf{x})\| _2$  and the third column shows the average of the same quantity normalized by the norm of the input gradient,  $\mathcal{C}_f(\mathbf{x}) = \| \nabla^2 f(\mathbf{x})\| _2 / \| \nabla f(\mathbf{x})\| _2$  The standard deviation is denoted by numbers enclosed in brackets.  

<table><tr><td>Model</td><td>E_x||∇^2f(x)||_2</td><td>E_xC_f(x)</td></tr><tr><td>Standard</td><td>600.06 (29.76)</td><td>73.99 (6.62)</td></tr><tr><td>DDN AT</td><td>2.86 (1.22)</td><td>4.32 (2.91)</td></tr><tr><td>SDF AT (Ours)</td><td>0.73 (0.08)</td><td>1.66 (0.86)</td></tr></table>

models. In particular, FAB, DDN, and FMN's performance degrades when the dataset changes. In contrast, SDF achieves smaller perturbations at a significantly lower cost than ALMA. This shows that the geometric interpretation of optimal adversarial perturbation, rather than viewing (1) as a non-convex optimization problem, can lead to an efficient solution. On the complexity aspect, the proposed approach is substantially faster than the other methods. In contrast, these approaches involve a costly minimization of a series of objective functions. We empirically observed that SDF converges in less than 5 or 6 iterations to a fooling perturbation; our observations show that SDF consistently achieves state-of-the-art minimum-norm perturbations across different datasets, models, and training strategies, while requiring the least number of gradient computations. This makes it readily suitable to be used as a baseline method to estimate the robustness of very deep neural networks on large datasets.

# 4.3 SDF ADVERSARIAL TRAINING (AT)

In this section, we evaluate the performance of a model adversarially trained using SDF against minimum-norm attacks and AutoAttack. Our experiments provide valuable insights into the effectiveness of adversarial training with SDF and sheds light on its potential applications in building more robust models. Adversarial training requires computationally efficient attacks, making costly options such as C&W unsuitable. Therefore, an attack that is parallelizable (both on batch size and gradient computation) is desired for successful adversarial training. SDF possesses these crucial properties, making it a promising candidate for building more robust models.

We adversarially train a WRN-28-10 on CIFAR10. Similar to the procedure followed in Rony et al. (2019), we restrict  $\ell_2$ -norms of perturbation to 2.6 and set the maximum number of iterations for SDF to 6. We train the model on clean examples for the first 200 epochs, and we then fine-tune it with SDF generated adversarial examples for 60 more epochs. Since a model trained using DDN-generated samples Rony et al. (2019) has demonstrated greater robustness compared to a model trained using PGD Madry et al. (2017), we compare our model against the former. Our model reaches a test

accuracy of  $90.8\%$  while the model by Rony et al. (2019) obtains  $89.0\%$ . SDF adversarially trained model does not overfit to SDF attack because, as Table 6 shows, SDF obtains the smallest perturbation. It is evident that SDF adversarially trained model can significantly improve the robustness of model against minimum-norm attacks up to  $30\%$ . In terms of comparison of these two adversarially trained models with AutoAttack (AA), our model outperformed the Rony et al. (2019) by improving about  $8.4\%$  against  $\ell_{\infty}$ -AA, for  $\varepsilon = 8/255$ , and  $0.6\%$  against  $\ell_{2}$ -AA, for  $\varepsilon = 0.5$ .

Furthermore, compared to a network trained on DDN samples, our adversarially trained model has a smaller input curvature (Table 7). This observation corroborates the idea that a more robust network will exhibit a smaller input curvature Moosavi-Dezfooli et al. (2019); Srinivas et al.; Qin et al. (2019).

# 4.4 AUTOATTACK++

Table 8: Analysis of robust accuracy for various defense strategies against AA++ and AA with  $\varepsilon = 0.5$  for six adversarially trained models on CIFAR-10. All models are taken from the RobustBench library Croce et al. (2020).  

<table><tr><td rowspan="2">Models</td><td rowspan="2">Clean acc.</td><td colspan="2">AA</td><td colspan="2">AA++</td></tr><tr><td>Robust acc.</td><td>Grads</td><td>Robust acc.</td><td>Grads</td></tr><tr><td>R1 Rebuffi et al. (2021)</td><td>95.7%</td><td>82.3%</td><td>1259.2</td><td>82.1%</td><td>599.5</td></tr><tr><td>R2 Sehwag et al. (2021)</td><td>90.3%</td><td>76.1%</td><td>1469.1</td><td>76.1%</td><td>667.7</td></tr><tr><td>R3 Gowal et al. (2020)</td><td>89.4%</td><td>63.4%</td><td>1240.4</td><td>62.2%</td><td>431.5</td></tr><tr><td>R4 Rice et al. (2020)</td><td>88.6%</td><td>67.6%</td><td>933.7</td><td>68.4%</td><td>715.3</td></tr><tr><td>R5 Rice et al. (2020)</td><td>89.05%</td><td>66.4%</td><td>846.3</td><td>62.5%</td><td>613.7</td></tr><tr><td>R6 Ding et al. (2018)</td><td>88.02%</td><td>67.6%</td><td>721.4</td><td>63.4%</td><td>511.1</td></tr><tr><td>Standard trained</td><td>94.7%</td><td>0.00%</td><td>208.6</td><td>0.00</td><td>121.1</td></tr></table>

In this part, we introduce a new variant of AutoAttack by introducing AutoAttack++ (AA++.). AutoAttack (AA) is a reliable and powerful ensemble attack that contains three types of white-box and a strong black-box attacks. AA evaluates the robustness of a trained model to adversarial perturbations whose  $\ell_2 / \ell_{\infty}$ -norm is bounded by  $\varepsilon$ . By substituting SDF with the attacks in the AA, we significantly increase the performance of AA in terms of computational time. Since SDF is an  $\ell_2$ -norm attack, we use the  $\ell_2$ -norm version of AA as well. We restrict maximum iterations of SDF to 10. If the norm of perturbations exceeds  $\varepsilon$ , we renormalize the perturbation to ensure its norm stays  $\leq \varepsilon$ . In this context, we have modified the AA algorithm by replacing  $\mathrm{APGD}^{\top}$  Croce & Hein (2020b) with SDF due to the former's cost and computation bottleneck in the context of AA. We compare the fooling rate and computational time of AA++ and AA on the stat-of-the-art models from the RobustBench leaderboard. In Table 8, we observe that AA++ is up to three times faster than AA. In an alternative scenario, we added the SDF to the beginning of the AA set, resulting in a version that is up to two times faster than the original AA, despite now containing five attacks (see Appendix). This outcome highlights the efficacy of SDF in finding adversarial examples. These experiments suggest that leveraging efficient minimum-norm and non-fixed iteration attacks, such as SDF, can enable faster and more reliable evaluation of the robustness of deep models.

# 5 CONCLUSION

In this work, we have introduced a family of parameter-free, fast, and parallelizable algorithms for crafting optimal adversarial perturbations. Our proposed algorithm, SDF, outperforms state-of-the-art  $\ell_2$ -norm attacks, while maintaining a small computational cost. We have demonstrated its effectiveness in various scenarios. Furthermore, we have shown that adversarial training using the examples generated by SDF builds more robust models. While our primary focus in this work has been on minimal  $\ell_2$  attacks, there exists potential for extending SDF families to other threat models, including general  $\ell_p$ -norms and targeted attacks. In the Appendix, we have demonstrated straightforward modifications that highlight the applicability of SDF to both targeted and  $\ell_{\infty}$ -norm attacks. However, a more comprehensive evaluation remains a direction for future work. Moreover, further limitations of our proposed method are elaborated upon in Appendix O.

# REFERENCES

Maximilian Augustin, Alexander Meinke, and Matthias Hein. Adversarial robustness on in-and out-distribution improves explainability. In European Conference on Computer Vision, pp. 228-245. Springer, 2020.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017, IEEE symposium on security and privacy (sp), pp. 39-57. IEEE, 2017.  
Nicholas Carlini, Anish Athalye, Nicolas Papernot, Wieland Brendel, Jonas Rauber, Dimitris Tsipras, Ian Goodfellow, Aleksander Madry, and Alexey Kurakin. On evaluating adversarial robustness. arXiv preprint arXiv:1902.06705, 2019.  
Jianbo Chen, Michael I Jordan, and Martin J Wainwright. Hopskipjumpattack: A query-efficient decision-based attack. In 2020 IEEE symposium on security and privacy (sp), pp. 1277-1294. IEEE, 2020.  
Francesco Croce and Matthias Hein. Minimally distorted adversarial examples with a fast adaptive boundary attack. In International Conference on Machine Learning, pp. 2196-2205. PMLR, 2020a.  
Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning, pp. 2206-2216. PMLR, 2020b.  
Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670, 2020.  
Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. Advances in neural information processing systems, 27, 2014.  
Gavin Weiguang Ding, Yash Sharma, Kry Yik Chau Lui, and Ruitong Huang. Mma training: Direct input space margin maximization through adversarial training. arXiv preprint arXiv:1812.02637, 2018.  
Logan Engstrom, Andrew Ilyas, Hadi Salman, Shibani Santurkar, and Dimitris Tsipras. Robustness (python library), 2019. URL https://github.com/MadryLab/robustness.  
Alhussein Fawzi, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. The robustness of deep networks: A geometrical perspective. IEEE Signal Processing Magazine, 34(6):50-62, 2017.  
Alhussein Fawzi, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard, and Stefano Soatto. Empirical study of the topology and geometry of deep networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3762-3770, 2018.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Sven Gowal, Chongli Qin, Jonathan Uesato, Timothy Mann, and Pushmeet Kohli. Uncovering the limits of adversarial training against norm-bounded adversarial examples. arXiv preprint arXiv:2010.03593, 2020.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pp. 630-645. Springer, 2016b.  
Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84-90, 2017.  
Yann LeCun, Patrick Haffner, Léon Bottou, and Yoshua Bengio. Object recognition with gradient-based learning. In Shape, contour and grouping in computer vision, pp. 319-345. Springer, 1999.  
Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3431-3440, 2015.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Tomáš Mikolov, Anoop Deoras, Daniel Povey, Lukáš Burget, and Jan Černocký. Strategies for training large scale neural network language models. In 2011 IEEE Workshop on Automatic Speech Recognition & Understanding, pp. 196-201. IEEE, 2011.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: A simple and accurate method to fool deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Jonathan Uesato, and Pascal Frossard. Robustness via curvature regularization, and vice versa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9078-9086, 2019.  
Maura Pintor, Fabio Roli, Wieland Brendel, and Battista Biggio. Fast minimum-norm adversarial attacks through adaptive norm constraints. Advances in Neural Information Processing Systems, 34:20052-20062, 2021.  
Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. Advances in neural information processing systems, 29, 2016.  
Chongli Qin, James Martens, Sven Gowal, Dilip Krishnan, Krishnamurthy Dvijotham, Alhussein Fawzi, Soham De, Robert Stanforth, and Pushmeet Kohli. Adversarial robustness through local linearization. Advances in Neural Information Processing Systems, 32, 2019.  
Rahul Rade and Seyed-Mohsen Moosavi-Dezfooli. *Helper-based adversarial training: Reducing excessive margin to achieve a better accuracy vs. robustness trade-off*. In ICML 2021 Workshop on Adversarial Machine Learning, 2021.  
Ali Rahmati, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard, and Huaiyu Dai. Geoda: a geometric framework for black-box adversarial attacks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8446-8455, 2020.  
Sylvestre-Alvise Rebuffi, Sven Gowal, Dan A Calian, Florian Stimberg, Olivia Wiles, and Timothy Mann. Fixing data augmentation to improve adversarial robustness. arXiv preprint arXiv:2103.01946, 2021.  
Leslie Rice, Eric Wong, and Zico Kolter. Overfitting in adversarially robust deep learning. In International Conference on Machine Learning, pp. 8093-8104. PMLR, 2020.  
Jerome Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed, Robert Sabourin, and Eric Granger. Decoupling direction and norm for efficient gradient-based 12 adversarial attacks and defenses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.  
Jérôme Rony, Eric Granger, Marco Pedersoli, and Ismail Ben Ayed. Augmented lagrangian adversarial attacks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7738-7747, 2021.

Vikash Sehwag, Saeed Mahloujifar, Tinashe Handina, Sihui Dai, Chong Xiang, Mung Chiang, and Prateek Mittal. Improving adversarial robustness using proxy distributions. CoRR, abs/2104.09425, 2021.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, and François Fleuret. Efficient training of low-curvature neural networks. In Advances in Neural Information Processing Systems.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to adversarial example defenses. In Advances in Neural Information Processing Systems, 2020.  
Jonathan Uesato, Brendan O'donoghue, Pushmeet Kohli, and Aaron Oord. Adversarial risk and the dangers of evaluating against weak attacks. In International Conference on Machine Learning, pp. 5025-5034. PMLR, 2018.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.  
Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li, Duane Boning, and Cho-Jui Hsieh. Towards stable and efficient training of verifiably robust neural networks. arXiv preprint arXiv:1906.06316, 2019.
