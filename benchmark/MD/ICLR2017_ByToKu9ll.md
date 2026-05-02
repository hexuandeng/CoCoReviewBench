# EVALUATION OF DEFENSIVE METHODS FOR DNNS AGAINST MULTIPLE ADVERSARIAL EVASION MODELS

Xinyun Chen

Shanghai Jiao Tong University

jungyhuk@gmail.com

Bo Li

University of Michigan

bbbli@umich.edu

Yevgeniy Vorobeychik

Vanderbilt University

yevgeniy.vorobeychik@vanderbilt.edu

# ABSTRACT

Due to deep cascades of nonlinear units, deep neural networks (DNNs) can automatically learn non-local generalization priors from data and have achieved high performance in various applications. However, such properties have also opened a door for adversaries to generate the so-called adversarial examples to fool DNNs. Specifically, adversaries can inject small perturbations to the input data and therefore decrease the performance of deep neural networks significantly. Even worse, these adversarial examples have the transferability to attack a black-box model based on finite queries without knowledge of the target model. Therefore, we aim to empirically compare different defensive strategies against various adversary models and analyze the cross-model efficiency for these robust learners. We conclude that the adversarial retraining framework also has the transferability, which can defend adversarial examples without requiring prior knowledge of the adversary models. We compare the general adversarial retraining framework with the state-of-the-art robust deep neural networks, such as distillation, autoencoder stacked with classifier (AEC), and our improved version, IAEC, to evaluate their robustness as well as the vulnerability in terms of the distortion required to mislead the learner. Our experimental results show that the adversarial retraining framework can defend most of the adversarial examples notably and consistently without adding additional vulnerabilities or performance penalty to the original model.

# 1 INTRODUCTION

Despite the success of deep neural networks (DNNs) in diverse areas, ranging from image recognition and machine translation to autonomous driving, its vulnerabilities have been exploited in the adversarial environments. Evasion attacks against such deep learning systems have recently received considerable attention (Goodfellow et al., 2014; Papernot et al., 2016c; Nguyen et al., 2015; Szegedy et al., 2013). It has been shown that with small magnitude of noise added, the original instance can easily be misclassified by the otherwise accurate deep neural networks. Efforts have been made to understand such adversarial examples. Goodfellow et al. (2014) pointed out that the adversarial examples actually make use of the linear nature of the DNNs based on the observation of their generalization across architectures and training sets. Tabacof & Valle (2015) analyzed the adversarial image space and showed that adversarial images appear in large regions in the pixel space. Papernot et al. (2016c) studied the limitation of adversarial evasion examples and showed that some instances are more difficult to manipulate than the others. Sabour et al. (2015) demonstrated that the attacker can change classification to an arbitrary class by malicious manipulations. The reverse engineering problem has been proposed in Vorobeychik & Li (2014), and it theoretically proved that the black-box attack is possible and also showed one could learn a sufficiently similar classifier from queries both theoretically and empirically. Similarly, even without knowing exactly the learning algorithm, several black-box attacks have been proposed targeting DNNs, which demonstrates the transferability of such adversarial examples (Papernot et al., 2016a;b).

Given the strong evasion properties of these adversarial examples, some works have been proposed to test and investigate the robustness of the deep neural networks against the adversarial examples. Zheng et al. have proposed to stabilize the state-of-the-art Inception architecture against different distortions, while it focuses on general random noise or distortions, such as compression, rescaling and cropping on images, instead of the adversarial noise in Zheng et al. (2016). Miyato et al. (2015) have proposed to apply the local distribution smoothness for statistical model to promote the smoothness of the model distribution and conduct the virtual adversarial training to enhance the performance of deep neural networks. However, it did not test on the adversarial examples and still had a long way to perform robustly against these real adversarial instances. Several autoencoder structures have been proposed against the adversarial examples by reconstructing the original images ahead of classification Gu & Rigazio (2014); Vincent et al. (2008). Jin et al. have proposed a feedforward CNN structure to improve the robustness in the presence of adversarial noise, which is restricted to the specific type of models in Jin et al. (2015). Jan et al. (2002) has proposed to explore the perturbed regions and apply ensemble method to enhance the robustness of classification. However, the focus of these researches have been less on adversarial evasion modeling and more on robustness of learning to small perturbations (e.g., due to noise). This is in contrast to most prior literature on adversarial learning, where adversarial modeling explicitly considers adversaries as balancing evasion cost and success.

We focus on providing thorough analysis for different algorithmic strategic defensive learners against various adversary models considering their robustness against adversarial examples, efficiency for cross-model learning process, resilience against additional attacks, and the vulnerabilities of these learners. A nice symmetry analysis for both the adversary and learner is provided through these analyses. We show that the general adversarial retraining framework performs significantly robust compared with the state-of-the-art defensive algorithms. For example, even for the black box attack, which is considered hard to defend, as long as there is a way to generate these adversarial evasion examples, the robust adversarial restraining framework can always improve the learning ability without knowing the actual adversary model.

In summary, we made the following contributions:

1. Evaluate the robustness of the general robust adversarial retraining framework (RAD) with the state-of-the-art AEC, Distillation, and the improved AEC, against different adversary models;  
2. Propose an improved AutoEncoder stacked with Classifier (IAEC);  
3. Compare the cross-model learning efficiency of different defensive methods and demonstrate the ability to defend against black-box attacks;  
4. Demonstrate the robustness of the retraining framework  $RAD, AEC, IAEC$ , and Distillation, against new attacks by attacking these robust learners repeatedly;  
5. Analyze the vulnerabilities induced by different defensive strategies/models based on their tolerance of the malicious distortions required to mislead the classifier.

We illustrate the applicability and efficiency of different defensive strategies against various state-of-the-art adversary models based on both MNIST and CIFAR-10 datasets.

# 2 PROBLEM

To understand the phenomenon of adversarial examples in deep neural networks, we aim to analyze potential defending methods against different adversary models from various perspectives, such as the robustness of the learner itself, the cross-model generalization ability, the resilience against additional attacks, and the vulnerabilities in terms of the required distortion to attack the robust learner again. Let  $X \subseteq R^n$  represent the feature space, with  $n$  the number of features. For every instance  $x_i \in X$ , which is drawn from certain distribution  $x_i \sim D$ , there is a corresponding label  $y_i \in \mathcal{V}$  to comprise the data pair  $(x_i, y_i)$ , where  $x_{ij}$  denotes the  $j$ th feature of  $x_i$ .

In the adversarial environments, adversary would like to accomplish the goal of evading the classifier. To formalize, suppose that  $M \subseteq \mathcal{V}$  is a set of labels which an adversary wishes to attack, and let  $z(m)$  be the target label for each  $m \in M$ . For example, for autonomous driving, potential

adversaries may aim to manipulate a stop sign or a dead-end warning sign, to a lamppost, a tree, or an advertisement sign, to cause accidents. Since such perturbations on images towards deep neural networks are often imperceptible to human eyes, it can cause serious vulnerabilities when deploying the DNNs in real adversarial environments. The defender's goal is to learn a classifier with parameters  $w$ ,  $g_w: x_i \to \mathcal{Y}$ , using a training data set of labeled instance  $T = \{(x_1, y_1), \dots, (x_m, y_m)\}$ . Here, we focus on deep neural networks representing the function  $g_w(\cdot)$ . Therefore, the learner's objective is to minimize the following general loss function:

$$
\min  _ {w} \mathcal {L} (w; \mathcal {A}) = \sum_ {i: y _ {i} \in \mathcal {Y} \backslash M} l \left(g _ {w} \left(x _ {i}\right), y _ {i}\right) + \sum_ {i: y _ {i} \in M} l \left(g _ {w} \left(\mathcal {A} \left(w, x _ {i}\right), y _ {i}\right) + \alpha \| w \| _ {p} ^ {p}, \right. \tag {1}
$$

where  $l(\cdot)$  can be arbitrary loss function and  $\mathcal{A}$  represents the adversary model.

The adversarial risk function in Equation 1 is general: it can be any adversary model oracle,  $\mathcal{A}$ , which is used to generate the adversarial evasion instances. Traditionally, this adversarial oracle may capture evasion attack models based on minimizing evasion cost (Lowd & Meek, 2005; Li & Vorobeychik, 2014; Biggio et al., 2014), or based on actual attacker evasion behavior obtained from experimental data (Ke et al., 2016). More formally, we will discuss the potential adversary models for deep neural networks and the possible defensive models for the learner in detail below.

# 2.1 ADVERSARY MODEL

To mislead deep neural networks, various methods have been proposed to generate the adversarial examples. We mainly discuss three state-of-the-art adversary models  $\mathcal{A}$  here for further evaluation.

Fast Gradient Sign. Based on the linear view of adversarial examples, a fast way of generating these adversarial examples were proposed in Goodfellow et al. (2014). Suppose  $x_{i}$  is the original feature vector, based on adversary model  $\mathcal{A}(fgs)$ , we have  $x_{i}' = x_{i} + \eta$ , where  $\eta$  represents the perturbation added for the original instance. Therefore, the dot product between the weighted parameter vector  $w$  and an adversarial example  $x_{i}'$  becomes:

$$
w ^ {T} x _ {i} ^ {\prime} = w ^ {T} x _ {i} + w ^ {T} \eta .
$$

Let  $J(w, x_i, y_i)$  be the cost used to train the neural network. By linearizing the cost function around the current value of  $w$ , an optimal max-norm constrained perturbation is generated as

$$
\eta = \epsilon \operatorname {s i g n} \left(\nabla_ {x} J \left(w, x _ {i}, y _ {i}\right)\right),
$$

where the adversary can vary  $\epsilon$  to generate adversarial examples with different attacking abilities for different deep neural networks.

Coordinate Greedy. Another more general adversary model  $\mathcal{A}(cg)$  is the local search framework Coordinate Greedy  $(cg)$  proposed in Li et al. (2016) for approximating the optimal adversarial instance. As an illustration, we focus on binary classification, and assume that  $g_w(x) = \mathrm{sign}(f(x))$  for some continuous function  $f$ , which in this case would be represented by a deep neural network.

The coordinate greedy approach is quite general, but we consider a specific adversary objective in which the adversary here tries to balance between two considerations: 1) appear as benign as possible to the classifier, and 2) minimize the cost of modification of the original instance (e.g., minimally manipulate the image). Note that it is also natural to assume that the attacker obtains no value from a manipulation to the original feature vector if the result is still classified as malicious. Therefore, an adversary aiming to transform an instance  $x_{i}$  into an adversarial example  $x_{i}'$  aims to solve the following optimization problem:

$$
\min  _ {x _ {i} ^ {\prime} \in X} \min  \left\{0, f \left(x _ {i} ^ {\prime}\right) \right\} + c \left(x _ {i} ^ {\prime}, x _ {i}\right), \tag {2}
$$

where  $c(x_{i}^{\prime}, x_{i})$  is the cost function of modifying from  $x_{i}$  to  $x_{i}^{\prime}$ . Here  $c(x_{i}^{\prime}, x_{i}) \geq 0$ ,  $c(x_{i}^{\prime}, x_{i}) = 0$  iff  $x_{i}^{\prime} = x_{i}$ , and the cost function  $c$  is strictly increasing in  $\| x_{i}^{\prime} - x_{i} \|_{2}$  and strictly convex in  $x_{i}^{\prime}$ . Because Problem 2 is non-convex, so the objective of adversary can be formed to minimize an upper bound:

$$
\min  _ {x _ {i} ^ {\prime}} Q \left(x _ {i} ^ {\prime}\right) \equiv f \left(x _ {i} ^ {\prime}\right) + c \left(x _ {i} ^ {\prime}, x _ {i}\right). \tag {3}
$$

So the high-level idea of  $cg$  is to iteratively choose a feature, and greedily update this feature according to the partial derivatives of the attacker's objective as 3 to evade the classifier. Below, we

take the exponential cost function  $c(x_i', x_i) = \exp \left( \lambda (\sum_j (x_{ij'} - x_{ij})^2 + 1)^{1/2} \right)$  as an example to estimate the modification cost, which is also quite natural: options become exponentially less desirable to an attacker as they are more distant from their ideal attack. Then we take the following partial derivative to update the adversary's objective until the convergence.

$$
\frac {\partial Q (x _ {i} ^ {\prime})}{\partial x _ {i j}} = \frac {\partial f (x _ {i} ^ {\prime})}{\partial x _ {i j}} + \frac {\partial c (x _ {i} ^ {\prime} , x _ {i})}{\partial x _ {i j}} = \frac {\partial f (x _ {i} ^ {\prime})}{\partial x _ {i j}} + \frac {\lambda c (x _ {i} ^ {\prime} , x _ {i}) (x _ {i j} ^ {\prime} - x _ {i j})}{(\sum_ {j} (x _ {i j} ^ {\prime} - x _ {i j}) ^ {2} + 1) ^ {1 / 2}},
$$

To avoid the algorithm converges only to a locally optimal solution, random restarts strategy is applied to randomly select the starting points in the feature space. As long as a global optimum has a basin of attraction with positive Lebesgue measure, or the feature space is finite, this process will asymptotically converge to a globally optimal solution with enough random restarts.

Adam. Another adversary model  $\mathcal{A}(adam)$  applies the stochastic gradient-based optimization algorithm Adam to generate adversarial examples, which is for solving the first-order gradient-based optimization of stochastic objective functions based on adaptive estimates of lower-order moments in Kingma & Ba (2014). Let  $f(w)$  be the noisy objective function that is differentiable w.r.t. parameter vector  $w$ . With  $f_{1}(w), f_{w}(w), \ldots, f_{T}(w)$  the realizations of the stochastic function at subsequent timesteps 1, 2, ...,  $T$ . The stochasticity might come from the evaluation at random subsamples (mini-batches) of data points, or arise from inherent function noise. The gradient evaluated at timestep  $t$  w.r.t  $w$  is present as below

$$
g _ {t} = \nabla_ {w} f _ {t} (w).
$$

Therefore, the adversary can update the malicious instances by adding the malicious noise from different mini-batches until it evades the deep neural networks classifier.

# 2.2 DEFENDER MODEL

Given the possible adversary models, several defensive strategies have been proposed focusing on different perspectives. Basically, the learner tries to integrate the prior knowledge of either the adversary model or the data distribution with the classification process. Here we consider different defensive strategies given the adversary model and form the interaction as a Stackelberg game. We will also consider the repeated game setting in section 3.3.

Adversarial Retraining framework (RAD). A systematic defensive approach based on adversarial retraining (RAD) has been proposed in Li et al. (2016). At the high level, RAD starts with the original training data and iteratively updating the learner with adversarial instances that evade the previously computed classifier until the convergence. It has been proved that the algorithm will terminate and the lower bound of the empirical loss of RAD is also provided. The important part for RAD is to select the adversarial retraining instances. In practical, it is hard to exactly estimate the adversary model as well as the parameters used within their model. Therefore, the generalization ability of RAD across different adversary models are quite important. Surprisingly, RAD generalizes quite well among various adversary models without requiring to know the exact attacker strategy. We will present the cross-model analysis for RAD in details in section 3.2.

AutoEncoder stacked with Classifier (AEC). One of the recent and efficient defensive method is the AutoEncoder stacked with a classifier to initialize deep architectures proposed in Gu & Rigazio (2014). To assess the structure of the adversarial noise, a three-hiden-layer autoencoder on mapping adversarial examples back to the original data samples is trained and stacked with the classifier. We train the AutoEncoder with different adversarial algorithms, including the fast gradient sign method  $(fgs)$ , the coordinate greedy  $(cg)$  method, as well as Adam Kingma & Ba (2014).

Improved AutoEncoder stacked with Classifier (IAEC). Since the baseline AEC cannot perform very well by only mapping the adversarial images back to the original image, we apply an improved AutoEncoder stacked with classifier (IAEC) defensive method. As AutoEncoder itself can not ensure that adversarial examples are denoised, we add a cross-entropy regularizer term as the loss function to help ensure that the output of AutoEncoder is classified correctly. Let  $y_{i}$  be the one-hot representation of ground truth label of an input instance  $x_{i}$ , then our loss function becomes:

$$
J \left(x _ {i}\right) = \| s \left(x _ {i}\right) - x _ {i} ^ {\prime} \| + H \left(y _ {i}, f \left(x _ {i}\right)\right),
$$

where  $s(x_{i})$  represents the mapping result of  $x_{i}$  by the AutoEncoder, and the cross-entropy function  $H(y_{i},f(x_{i})) = -\sum_{x_{i}}y_{i}\log f(x_{i}).$

Distillation. Considering the fact that the knowledge extracted during training, which is in the form of probability vectors, and transferred in smaller networks to maintain accuracy comparable with those of larger networks can also be beneficial to improving generalization capabilities of deep neural networks outside of their training dataset, a defensive strategy against the adversarial examples has been proposed in Papernot et al. (2015). This defensive strategy transfers the knowledge contained in probability vectors through the distillation training step, then applies these probabilities in the next training step instead of using the original hard labels, and therefore enhances its resilience to perturbations. This defensive model is independent of the adversary models and we will evaluate its robustness and vulnerabilities in details in section 3.

# 3 EXPERIMENTAL ANALYSIS

In this section, we empirically compare the adversarial retraining framework  $RAD$  with the other state-of-the-art baseline method Distillation Papernot et al. (2015), AutoEncoder stacked with Classifier (AEC) Gu & Rigazio (2014) and our improved AutoEncoder stacked with Classifier (IAEC) against various adversary models based on both MNIST and CIFAR-10 datasets. Particularly, here we apply to stack the AutoEncoder with LeNet-5 LeCun et al. (1998).

Basically, we first analyze the robustness of  $RAD$  and Distillation, which performs the best against adversarial examples currently, by comparing the classification results before and after applying the adversarial retraining technique based on both MNIST and CIFAR-10 datasets. Then we estimate the cross-model classification robustness for  $RAD$ ,  $AEC$ , the improved  $IAEC$ , and Distillation. Precisely, during the cross-model evaluation, we allow the attacker to generate the adversarial examples with different adversarial algorithms, while the defender has no clue about what adversarial algorithm is used. Therefore, we are able to evaluate the resilience of the "black-box" defensive strategies without requiring to know the actual adversary model.

Besides, we allow the attacker to attack these robustly enhanced learners and we compare the resilience of the  $RAD$  with the baseline defensive models and show that with retraining instances generated by adam, the  $RAD$  is almost unassailable for attacks based on the fast gradient sign method, which is promising to design universal defensive algorithms based on  $RAD$ .

Additionally, another perspective to measure the robustness of the learners is to evaluate how much noise is needed to make the learner misclassify an otherwise correct instance. As pointed out by Gu & Rigazio (2014), even a learner can be demonstrated to perform robustly against certain adversarial examples, it may become more vulnerable in the sense of being attacked by adding much smaller magnitude of adversarial noise. This means increasing the noticeability of the smallest adversarial noise for each example becomes the key to solve the adversarial examples problem. Therefore, we compare the malicious distortion required to attack each model, aiming to evaluate the vulnerability of different learners. The distortion is measured by  $d(x_{i}^{\prime},x_{i}) = \frac{1}{n}\sqrt{\sum(x_{i}^{\prime} - x_{i})^{2}}$ , where  $x_{i}^{\prime} = \mathcal{A}(\beta ,x_{i})$  representing the adversarial manipulated instance based on arbitrary adversary model  $\mathcal{A}$ .

# 3.1 ROBUSTNESS ANALYSIS FOR DEFENSIVE LEARNERS

To evaluate the robustness and efficiency of the adversarial retraining framework and other defensive learners, we generate adversarial examples based on the the coordinate gradient algorithm  $(cg)$ , adam Kingma & Ba (2014), and the fast gradient sign algorithm  $(fgs_{\epsilon})$  with the size of perturbation  $\epsilon = 0.1 \sim 0.5$  (Goodfellow et al. (2014)), respectively. Figure 1 shows the analysis of recall for the traditional LeNet-5 and the robust RAD classifiers on MNIST. The test error of LeNet-5 on the original dataset is  $0.045\%$ . It is obvious that after the adversarial retraining process based on RAD, the classifiers perform nearly optimal. It is interesting to observe that with the  $\epsilon$  of  $fgs$  increases, the adversarial examples generated by  $fgs$  can attack the original LeNet-5 more efficiently.

Figure 2 presents the comparisons of recall for the original LeNet-5 and the adversarial retraining framework on CIFAR-10. It shows that the adversarial retraining framework works robustly against different adversarial example generation methods. Note the test error of LeNet-5 on the original dataset is  $5.5\%$ . From the results of recall, we can see that almost all the "generated" adversarial instances are correctly classified by the retraining framework. Additionally, sometimes the test error of RAD is even smaller than that of the original model LeNet-5 based on the uncontaminated (no adversary) data. This means, with the adversarial robust retraining process, some "blind-spots"

![](images/a4296c3eb40b1d6e44572d6764d8fd88b152b2c604c81ccd7f695421f410ee00.jpg)  
(a)

![](images/353264bbf16fd18add4ad438792afc168144f71850d4e9828013477ffb7642bf.jpg)  
(b)  
Figure 1: Performance of retraining with instances generated from different models based on MNIST. (a) The retraining instances are generated by  $cg$ ; (b) the adversarial examples are generated by  $cg$ ; (c) the adversarial examples are generated by adam.

![](images/b6aff762f15e40f238cab138c694d05546b44ddb1e164962d960eb2081bb7253.jpg)  
(c)

in the input space volume can be filled out without decreasing the performance on the normal test data. Moreover, surprisingly, with the increase of  $\epsilon$ , the fast gradient sign method works worse for generating adversarial examples against LeNet-5, which is different for MNIST. This is actually caused by the properties of the fast gradient sign method itself. By following the gradient, the generated instance can be trapped into sub-optimal and therefore fail to converge to the global optima, so different step size can affect their final convergence. Therefore, by comparing with the results of MNIST, we can see learners on CIFAR-10 is easier to be trapped by the sub-optima and larger  $\epsilon$  values can lead the learner to be trapped into these points with higher probability. On the other hand, no matter how much the strength of adversarial ability is affected by different parameters, the adversarial retraining framework works robustly by almost identifying all the manipulated instances correctly on different datasets consistently.

![](images/6696d0c4e9316b831148cbb5bccece5479c36683f7ba35cae6fd031d370d0d4e.jpg)  
(a)

![](images/204546d88126e002562adc2a6c2d2a266010365ed2bb8bfbd0184fb886ca9efc.jpg)  
(b)  
Figure 2: Performance of retraining with instances generated from different models based on CIFAR-10. (a) The retraining instances are generated by CG; (b) the adversarial examples are generated by CG; (c) the adversarial examples are generated by adam.

![](images/07ed7f20980a96a6c5dd30817bcda5d0d15bf87920ae95b2b2c4efca9943adae.jpg)  
(c)

# 3.2 CROSS-MODEL ANALYSIS FOR DIFFERENT DEFENSIVE LEARNERS

Aiming to defend a more broad class of attacks, here we assume the learner has no clue about which adversarial algorithm the attacker uses to generate the adversarial instances. Therefore, the defender can perform robustly as the "black-box" learner against arbitrary adversaries. Here we use different attack algorithms to generate the adversarial examples, and the retraining instances for  $RAD$  are also generated across various adversary models to evaluate the learners' generalization ability. We also compare the results with the state-of-the-art Distillation,  $AEC$  and our improved IAEC algorithm based on different adversarial models. Here the  $AEC$  is trained on the adam model, which offers the best classification results. The IAEC is also trained corresponding to different adversary models to compare the cross-model learning ability with  $RAD$ . Table 1 shows the test error comparisons for these cross-model learners. "No adversary" presents the test error of different learners on the clean data. Basically, the adversarial retraining framework performs consistently better than  $AEC$ , IAEC, and Distillation on all different adversarial examples in terms of the classification error. This conclusion is independent of what models are used to generate adversarial retraining instances for  $RAD$ . Based on the results, the adversarial retraining framework has the potential to be applied to defend against any arbitrary attacks without requiring to know the exact adversary model. Based on the classification error results for our improved IAEC in Table 1, it is obvious that the IAEC with the same adam adversary model works much more robust than  $AEC$ . This means the proposed IAEC

is much more robust compared with the original AEC by adding the cross-entropy regularization. Additionally, we also evaluate the cross-model classification error for IAEC to test its generalization ability. Table 1 shows that the IAEC can also defend against different adversarial examples without requiring to know the exact adversary model.

Table 1: Classification error of different learners against various adversary models based on MNIST  

<table><tr><td>Model</td><td>A(fgs0.1)</td><td>A(fgs0.5)</td><td>A(cg)</td><td>A(adam)</td><td>No adversary</td></tr><tr><td>LeNet-5</td><td>1.2%</td><td>46.1%</td><td>48.2%</td><td>48.9%</td><td>0.045%</td></tr><tr><td>RAD(fgs0.1)</td><td>0.1%</td><td>0.5%</td><td>0.4%</td><td>3.0%</td><td>0.045%</td></tr><tr><td>RAD(fgs0.5)</td><td>0.5%</td><td>0.1%</td><td>0</td><td>2.5%</td><td>0.045%</td></tr><tr><td>RAD(cg)</td><td>0.1%</td><td>1.4%</td><td>0.4%</td><td>2.9%</td><td>0.045%</td></tr><tr><td>RAD(adam)</td><td>0</td><td>0.1%</td><td>0.1%</td><td>0.1%</td><td>0.045%</td></tr><tr><td>AEC(adam)</td><td>3.2%</td><td>20.6%</td><td>9.7%</td><td>2.6%</td><td>4.5%</td></tr><tr><td>IAEC(fgs0.1)</td><td>1.3%</td><td>28.0%</td><td>18.3%</td><td>9.6%</td><td>1.1%</td></tr><tr><td>IAEC(fgs0.5)</td><td>1.2%</td><td>1.4%</td><td>2.6%</td><td>5.5%</td><td>1.0%</td></tr><tr><td>IAEC(cg)</td><td>1.6%</td><td>1.6%</td><td>1.5%</td><td>7.4%</td><td>1.2%</td></tr><tr><td>IAEC(adam)</td><td>1.2%</td><td>5.2%</td><td>7.3%</td><td>2.3%</td><td>1.7%</td></tr><tr><td>Distillation(T=1)</td><td>0.6%</td><td>47.2%</td><td>29.4%</td><td>41.9%</td><td>0.2%</td></tr><tr><td>Distillation(T=100)</td><td>0.3%</td><td>42.3%</td><td>12.4%</td><td>28.5%</td><td>0.2%</td></tr></table>

Similarly, we show the classification error comparison results of  $RAD$  across different adversary models in Table 2 compared with Distillation. As CIFAR-10 images are more complex, the error rates for adversarial retraining framework get larger compared with that on MNIST. However, overall the classification error for the retraining framework on different adversarial examples are below  $13\%$  with zero knowledge of the adversary model, while the classification error on normal data is around  $6\%$ . Therefore, even on CIFAR-10 dataset, the adversarial retraining framework is still promising to perform the "black-box" defending resiliently against various attacks. Additionally, the distillation with  $T = 1$  and  $T = 100$  both encounter higher test error than  $RAD$ , even the distillation method performs more robustly when  $T = 100$  than  $T = 1$ .

Table 2: Comparisons for the error rate of  ${RAD}$  based on different adversary models on CIFAR-10  

<table><tr><td>Model</td><td>A(fgs0.1)</td><td>A(fgs0.5)</td><td>A(cg)</td><td>A(adam)</td><td>No adversary</td></tr><tr><td>LeNet-5</td><td>1.2%</td><td>46.1%</td><td>54.0%</td><td>52.7%</td><td>5.5%</td></tr><tr><td>RAD(fgs0.1)</td><td>2.35%</td><td>2.0%</td><td>4.65%</td><td>3.0%</td><td>5.3%</td></tr><tr><td>RAD(fgs0.5)</td><td>4.4%</td><td>2.7%</td><td>5.6%</td><td>2.6%</td><td>5.8%</td></tr><tr><td>RAD(cg)</td><td>7.5%</td><td>2.45%</td><td>5.05%</td><td>2.2%</td><td>5.7%</td></tr><tr><td>RAD(adam)</td><td>16.2%</td><td>2.8%</td><td>6.15%</td><td>2.4%</td><td>5.9%</td></tr><tr><td>Distillation(T=1)</td><td>21.3%</td><td>30.8%</td><td>13.8%</td><td>22.0%</td><td>11.0%</td></tr><tr><td>Distillation(T=100)</td><td>19.3%</td><td>25.2%</td><td>9.2%</td><td>20.2%</td><td>7.2%</td></tr></table>

# 3.3 ROBUSTNESS AGAINST ADDITIONAL ATTACKS

In order to test the robustness of the learner against the repeated attacks, where the attacker can again conduct attacks on the robust learners, here we evaluate how the robust learner behaves given additional attacks based on different adversary models. Table 3 presents the test error rate comparison when the attacker generates adversarial examples to attack the robust RAD learner, IAEC, and Distillation on MNIST. It is shown that the coordinate greedy  $(cg)$  and adam are somehow efficient to attack RAD, while the fast gradient sign methods fail to attack the robust RAD. So if the RAD is retrained with instances generated by arbitrary adversary models, it can be resilient against adversarial examples produced by the fast gradient sign method with various  $\epsilon$  values. This means the RAD can confer robustness to single-step attack methods but not the iterative ones. However, adversaries based on  $cg$  and adam can still find the vulnerabilities to attack the model. Compared with the performance of the adversarial retraining framework (RAD) against these "repeated attacks", the IAEC encounters much higher classification error when being attacked. This indicates that the adversarial

retraining framework can not only enhance the resilience of the original learner (LeNet-5), but also perform robustly against the additional attacks compared with the IAEC.

Table 3: Error rate of attacking the robust learners with additional attacks on MNIST  

<table><tr><td>Model</td><td>A(fgs0.1)</td><td>A(fgs0.5)</td><td>A(cg)</td><td>A(adam)</td></tr><tr><td>RAD(fgs0.1)</td><td>0.3%</td><td>9.6%</td><td>48.1%</td><td>49.0%</td></tr><tr><td>RAD(fgs0.5)</td><td>0.8%</td><td>0.1%</td><td>45.7%</td><td>49.0%</td></tr><tr><td>RAD(cg)</td><td>0.8%</td><td>3.4%</td><td>44.6%</td><td>49.0%</td></tr><tr><td>RAD(adam)</td><td>0.1%</td><td>0.1%</td><td>40.2%</td><td>48.7%</td></tr><tr><td>IAEC(fgs0.1)</td><td>4.2%</td><td>10.3%</td><td>49.9%</td><td>49.5%</td></tr><tr><td>IAEC(fgs0.5)</td><td>5.2%</td><td>3.8%</td><td>49.8%</td><td>49.9%</td></tr><tr><td>IAEC(cg)</td><td>5.3%</td><td>3.9%</td><td>49.9%</td><td>49.4%</td></tr><tr><td>IAEC(adam)</td><td>4.6%</td><td>7.0%</td><td>49.9%</td><td>49.9%</td></tr><tr><td>Distillation(T=100)</td><td>0.2%</td><td>0.2%</td><td>49.0%</td><td>48.7%</td></tr></table>

Table 4: Error rate of attacking the robust learners with additional attacks on CIFAR-10  

<table><tr><td>Model</td><td>A(fgs0.1)</td><td>A(fgs0.5)</td><td>A(cg)</td><td>A(adam)</td></tr><tr><td>RAD(fgs0.1)</td><td>3.7%</td><td>2.7%</td><td>42.0%</td><td>52.7%</td></tr><tr><td>RAD(fgs0.5)</td><td>5.3%</td><td>2.8%</td><td>49.0%</td><td>52.4%</td></tr><tr><td>RAD(cg)</td><td>7.9%</td><td>2.8%</td><td>52.0%</td><td>52.7%</td></tr><tr><td>RAD(adam)</td><td>6.3%</td><td>3.1%</td><td>54.0%</td><td>52.7%</td></tr><tr><td>Distillation(T=100)</td><td>9.05%</td><td>8.6%</td><td>54.0%</td><td>54.1%</td></tr></table>

Similarly, Table 4 presents the test error for attacking different robust learners with various adversary models on CIFAR-10.  $RAD$  again produces lower test error compared with Distillation ( $T = 1$ ,  $T = 100$ ) given diverse adversarial attacking strategies. What is worth to mention is that these robust learners all perform accurately on the normal dataset without adversarial manipulation, which offers more potentials for the robust learners.

# 3.4 VULNERABILITY OF THE DEFENSIVE LEARNERS

Given the fact that the attacker can attack the learning model continuously, here we are concerned with how vulnerable the robust models become in terms of the amount of distortion needed to add to mislead the learner. We compare the average distortion for attacking the LeNet-5, RAD, IAEC, and Distillation to evaluate their robustness. As mentioned by Gu & Rigazio (2014), AEC demands smaller distortion to attack, which means AEC is quite fragile, and we also gain the similar observation and confirm that attacking the original LeNet-5 model requires larger magnitude of noise than AEC. Thus, we focus on the improved IAEC.

In the Table 5 we present the demanded distortion to maliciously attack the RAD, the IAEC, and Distillation on MNIST. Note that the fast gradient sign method here is a one-step method, which will stop after computing one gradient to find the optimal perturbation of a linear approximation of the cost or model, so it cannot guarantee to find the evasion instance  $x_{i}^{\prime}$  and we do not consider its distortion. So here we only consider  $cg$  and adam to generate distortions. We use  $RAD(.)$  to represent the adversarial retraining framework retrained with arbitrary adversarial instances since they all require the same amount of distortion to be attacked given their similar network structures. From Table 5 RAD requires the same distortion as attacking the original LeNet-5 model. However, the distortion needed for attacking the IAEC is substantially smaller than that for attacking the original models. From this perspective, the IAEC becomes more vulnerable compared with the original model even though it can be resilient against the adversarial examples. Similar for Distillation, smaller distortion is demanded to attack the robust learner, which means more vulnerabilities are introduced by the robust Distillation. On the contrary, the adversarial retraining framework RAD can perform robustly against various diverse adversarial attacks without increasing the vulnerability penalty.

Table 5: Adversarial distortion required for attacking different models on MNIST  

<table><tr><td>Model</td><td>A(cg)</td><td>A(adam)</td></tr><tr><td>LeNet-5</td><td>0.0118</td><td>0.0060</td></tr><tr><td>RAD(.)</td><td>0.0118</td><td>0.0060</td></tr><tr><td>IAEC(fgs0.1)</td><td>0.0042</td><td>0.0031</td></tr><tr><td>IAEC(fgs0.5)</td><td>0.0058</td><td>0.0028</td></tr><tr><td>IAEC(cg)</td><td>0.0069</td><td>0.0023</td></tr><tr><td>IAEC(adam)</td><td>0.0064</td><td>0.0029</td></tr><tr><td>Distillation(T=100)</td><td>0.0106</td><td>0.0060</td></tr></table>

Figure 3 shows the results of adding the corresponding adversarial noise to generate the misclassification for LeNet-5 model by different adversarial algorithms qualitatively. It shows that by using fast gradient sign method with  $\epsilon = 0.5$ , the original image is almost distorted. This indicates different adversary models have different attacking strengths, so taking the stronger adversary model into account may have a chance to defend the weaker adversaries, which makes the universal defensive model promising.

![](images/d32a7f83970656d6767d9867736cb12a258e8ec9e5fe2e179b10f256fcaa1273.jpg)  
(a)

![](images/a17611dfcec0966148ae7add528a792833ee355a53ea491a9022a11e1c6ae0ab.jpg)  
(b)  
Figure 3: Visualization of adversarial examples generated by different attacker models based on MNIST. (a) Original image, (b) attacked by  $fgs_{0.1}$ , (c) attacked by  $fg s_{0.5}$ , (d) attacked by  $cg$ , (e) attacked by adam.

![](images/d61474e887d5223e4513f9e357f79f507de4e8db0d78c2663813a91cc7c62d10.jpg)  
(c)

![](images/b229b0cd4a4c5b5b40cded20b508846e873be326195958c3029b9135f24fec83.jpg)  
(d)

![](images/e22ff9e322811c1573221a6142bcc0e4f2cfa1035fcfcecec1045233c3c7c10a.jpg)  
(e)

Similarly, Table 6 lists the amount of distortion needed to fool the original learner based on CIFAR-10. It is shown that both the  $RAD$  and Distillation need exactly the same amount of distortion with the original LeNet-5 model, which means these robust learners do not increase the vulnerability of the original model.

Table 6: Adversarial distortion required for attacking different models on CIFAR-10  

<table><tr><td>Model</td><td>A(cg)</td><td>A(adam)</td></tr><tr><td>LeNet-5</td><td>0.0025</td><td>0.0015</td></tr><tr><td>RAD(.)</td><td>0.0025</td><td>0.0015</td></tr><tr><td>Distillation(T=100)</td><td>0.0025</td><td>0.0015</td></tr></table>

The visual attacking results by injecting malicious noise are shown in Figure 4. It is clear that  $fgs$  with  $\epsilon = 0.5$  can distort the original images the most compared with other adversary algorithms. Surprisingly, all the retraining framework based on different retraining instances only get the classification error lower than  $3.0\%$ .

![](images/4e0a971ed254865a782a05a59a223564e28f6708c689ba193d3d954824ca8bd1.jpg)  
(a)  
Figure 4: Visualization of adversarial examples generated by different attacker models based on CIFAR-10. (a) Original image, (b) attacked by  $fgs_{0.1}$ , (c) attacked by  $fg s_{0.5}$ , (d) attacked by  $cg$ , (e) attacked by adam.

![](images/6622cbdff68d077fb4966e428ff384baba420492b26fe82b80cfcd31eac2c8ae.jpg)  
(b)

![](images/2e53a4d88e77bcc28ed64cf58706eec2e750779ed9b44d4b659e7760934249df.jpg)  
(c)

![](images/20a395fad31253737e5ebf508add71620ffa7048190fb6a5417dd92e2042e912.jpg)  
(d)

![](images/e005d27a90bb1d5227e3bd1a5a7f02eb3fbc20978749faf957804900105d8923.jpg)  
(e)

# 4 CONCLUSION

To understand the adversarial examples better, as well as the potential adversary models and corresponding defensive learners, we conduct extensive experiments to evaluate properties of different defensive strategies. We point out that  $RAD$  works the best among all the defensive strategies against different adversary models, including one-step and iterative ones, in terms of the classification test error. The adversarial retraining framework,  $RAD$ , also generalizes well for the cross-model evaluation compared with  $AEC$ ,  $IAEC$ , and Distillation. Moreover, both  $RAD$  and Distillation do not introduce additional vulnerability penalty to the original models, while still increase the robustness. So in the future work, to generalize the robust learner across different adversary models, one direction could be to generate retraining instances based on diverse adversarial algorithms to cover as much as possible the "blind-spots" within the input space. In addition, we will dynamically optimize the choice of adversary model and the quantity of retraining instances according to the robustness requirements of a specific learner. Therefore, the tradeoff between robustness and accuracy on the normal data can be balanced based on the specific resilience demand of the learner.

# REFERENCES

Battista Biggio, Giorgio Fumera, and Fabio Roli. Security evaluation of pattern classifiers under attack. Knowledge and Data Engineering, IEEE Transactions on, 26(4):984-996, 2014.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Shixiang Gu and Luca Rigazio. Towards deep neural network architectures robust to adversarial examples. arXiv preprint arXiv:1412.5068, 2014.  
JC Jan, Shih-Lin Hung, SY Chi, and JC Chern. Neural network forecast model in deep excavation. Journal of Computing in Civil Engineering, 16(1):59-65, 2002.  
Jonghoon Jin, Aysegul Dundar, and Eugenio Culurciello. Robust convolutional neural networks under adversarial noise. arXiv preprint arXiv:1511.06306, 2015.  
Liyiming Ke, Bo Li, and Yevgeniy Vorobeychik. Behavioral experiments in email filter evasion. In AAAI Conference on Artificial Intelligence, 2016.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Bo Li and Yevgeniy Vorobeychik. Feature cross-substitution in adversarial classification. In Advances in Neural Information Processing Systems, pp. 2087-2095, 2014.  
Bo Li, Yevgeniy Vorobeychik, and Xinyun Chen. A general retraining framework for scalable adversarial classification. arXiv preprint arXiv:1604.02606, 2016.  
Daniel Lowd and Christopher Meek. Adversarial learning. In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining, pp. 641-647. ACM, 2005.  
Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, and Shin Ishii. Distributional smoothing with virtual adversarial training. stat, 1050:25, 2015.  
Anh Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for recognizable images. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 427-436. IEEE, 2015.  
Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation as a defense to adversarial perturbations against deep neural networks. arXiv preprint arXiv:1511.04508, 2015.

Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. Transferability in machine learning: from phenomena to black-box attacks using adversarial samples. arXiv preprint arXiv:1605.07277, 2016a.  
Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. Practical black-box attacks against deep learning systems using adversarial examples. arXiv preprint arXiv:1602.02697, 2016b.  
Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In 2016 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 372-387. IEEE, 2016c.  
Sara Sabour, Yanshuai Cao, Fartash Faghri, and David J Fleet. Adversarial manipulation of deep representations. arXiv preprint arXiv:1511.05122, 2015.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
Pedro Tabacof and Eduardo Valle. Exploring the space of adversarial images. arXiv preprint arXiv:1510.05328, 2015.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pp. 1096-1103. ACM, 2008.  
Yevgeniy Vorobeychik and Bo Li. Optimal randomized classification in adversarial settings. In Proceedings of the 2014 international conference on Autonomous agents and multi-agent systems, pp. 485-492. International Foundation for Autonomous Agents and Multiagent Systems, 2014.  
Stephan Zheng, Yang Song, Thomas Leung, and Ian Goodfellow. Improving the robustness of deep neural networks via stability training. arXiv preprint arXiv:1604.04326, 2016.