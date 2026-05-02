# TOWARDS DEEP LEARNING MODELS RESISTANT TO ADVERSARIAL ATTACKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent work has demonstrated that neural networks are vulnerable to adversarial examples, i.e., inputs that are almost indistinguishable from natural data and yet classified incorrectly by the network. To address this problem, we study the adversarial robustness of neural networks through the lens of robust optimization. This approach provides us with a broad and unifying view on much prior work on this topic. Its principled nature also enables us to identify methods for both training and attacking neural networks that are reliable and, in a certain sense, universal. In particular, they specify a concrete security guarantee that would protect against a well-defined class of adversaries. These methods let us train networks with significantly improved resistance to a wide range of adversarial attacks. They also suggest robustness against a first-order adversary as a natural and broad security guarantee. We believe that robustness against such well-defined classes of adversaries is an important stepping stone towards fully resistant deep learning models.

# 1 INTRODUCTION

Recent breakthroughs in computer vision and speech recognition are bringing trained classifiers into the center of security-critical systems. Important examples include vision for autonomous cars, face recognition, and malware detection. These developments make security aspects of machine learning increasingly important. In particular, resistance to adversarially chosen inputs is becoming a crucial design goal. While trained models tend to be very effective in classifying benign inputs, recent work (Dalvi et al., 2004; Szegedy et al., 2013; Goodfellow et al., 2014; Nguyen et al., 2015; Sharif et al., 2016) shows that an adversary is often able to manipulate the input so that the model produces an incorrect output.

This phenomenon has received particular attention in the context of deep neural networks, and there is now a quickly growing body of work on this topic (Fawzi et al., 2015; Kurakin et al., 2016; Papernot & McDaniel, 2016; Rozsa et al., 2016; Torkamani, 2016; Sokolic et al., 2016; Tramér et al., 2017b). Computer vision presents a particularly striking challenge: very small changes to the input image can fool state-of-the-art neural networks with high probability (Szegedy et al., 2013; Goodfellow et al., 2014; Nguyen et al., 2015; Sharif et al., 2016; Moosavi-Dezfooli et al., 2016). This holds even when the benign example was classified correctly, and the change is imperceptible to a human. Apart from the security implications, this phenomenon also demonstrates that our current models are not learning the underlying concepts in a robust manner. All these findings raise a fundamental question:

How can we learn models robust to adversarial inputs?

There are now many proposed defense mechanisms for the adversarial setting. Examples include defensive distillation (Papernot et al., 2016a; Papernot & McDaniel, 2016), feature squeezing (Xu et al., 2017), and several detection approaches for adversarial inputs (see Carlini & Wagner (2017) for references). While these works constitute important first steps in exploring the realm of possibilities, they do not offer a good understanding of the guarantees they provide. We can never be certain that a particular defense mechanism prevents the existence of some well-defined class of adversarial attacks. This makes it difficult to navigate the landscape of adversarial robustness or to fully evaluate the possible security implications. Moreover, subsequent work (Carlini & Wagner, 2016a; He et al., 2017)

has shown that most of these defenses can be bypassed by stronger, adaptive adversaries. (Further discussion of related work has been deferred to Section 6.)

In this paper, we study the adversarial robustness of neural networks through the lens of robust optimization. We use a natural saddle point (min-max) formulation to capture the notion of security against adversarial attacks in a principled manner. This formulation allows us to be precise about the type of security guarantee we would like to achieve, i.e., the broad class of attacks we want to be resistant to (in contrast to defending only against specific known attacks). The formulation also enables us to cast both attacks and defenses into a common theoretical framework. Most prior work on adversarial examples naturally fits into this framework. In particular, adversarial training directly corresponds to optimizing this saddle point problem. Similarly, prior methods for attacking neural networks correspond to specific algorithms for solving the underlying constrained optimization problem.

Equipped with this perspective, we make the following contributions.

1. We conduct a careful experimental study of the optimization landscape corresponding to this saddle point formulation. Despite the non-convexity and non-concavity of its constituent parts, we find that the underlying optimization problem is tractable after all. In particular, we provide strong evidence that first-order methods can reliably solve this problem and motivate projected gradient descent (PGD) as a universal "first-order adversary", i.e., the strongest attack utilizing the local first order information about the network. We supplement these insights with ideas from real analysis to further motivate adversarial training against a PGD adversary as a strong and natural defense.  
2. We explore the impact of network architecture on adversarial robustness and find that model capacity plays an important role. To reliably withstand strong adversarial attacks, networks require a significantly larger capacity than for correctly classifying benign examples only. This shows that a robust decision boundary of the saddle point problem can be significantly more complicated than a decision boundary that simply separates the benign data points.  
3. Building on the above insights, we train networks on MNIST and CIFAR10 that are robust to a wide range of adversarial attacks against adversaries bounded by 0.3 and 8 in  $\ell_{\infty}$  norm respectively. Our approach is based on optimizing the aforementioned saddle point formulation and uses our optimal "first-order adversary". Our best MNIST model achieves an accuracy of more than  $89\%$  against the strongest adversaries in our test suite. In particular, our MNIST network is even robust against white box attacks of an iterative adversary. Our CIFAR10 model achieves an accuracy of  $46\%$  against the same adversary. Furthermore, in case of the weaker black box (transfer) attacks, our MNIST and CIFAR10 networks achieve an accuracy of more than  $95\%$  and  $64\%$ , respectively (a more detailed overview can be found in Tables 1 and 2). To the best of our knowledge, we are the first to achieve these levels of robustness on MNIST and CIFAR10 against a broad set of attacks.

Overall, these findings suggest that secure neural networks are within reach. In order to further support this claim, we have invited the community to attempt attacks against our MNIST and CIFAR10 networks in the form of an open challenge. At the time of writing, we received about fifteen submissions to the MNIST challenge and the best submission achieved roughly  $93\%$  accuracy in a black box attack. We received no submissions for the CIFAR10 challenge that went beyond the  $64\%$  accuracy of our attack. Considering that other proposed defenses were often quickly broken (Carlini & Wagner, 2017), we believe that our robust models are significant progress on the defense side. Furthermore, recent work on verifiable adversarial examples showed that our proposed defense reliably increased the robustness to any  $\ell_{\infty}$ -bounded attack. $^{1}$

# 2 AN OPTIMIZATION VIEW ON ADVERSARIAL ROBUSTNESS

Much of our discussion will revolve around an optimization view of adversarial robustness. This perspective not only captures the phenomena we want to study in a precise manner, but will also inform our investigations. To this end, let us consider a standard classification task with an underlying

data distribution  $\mathcal{D}$  over pairs of examples  $x\in \mathbb{R}^d$  and corresponding labels  $y\in [k]$ . We also assume that we are given a suitable loss function  $L(\theta ,x,y)$ , for instance the cross-entropy loss for a neural network. As usual,  $\theta \in \mathbb{R}^p$  is the set of model parameters. Our goal then is to find model parameters  $\theta$  that minimize the risk  $\mathbb{E}_{(x,y)\sim \mathcal{D}}[L(x,y,\theta)]$ .

Empirical risk minimization (ERM) has been tremendously successful as a recipe for finding classifiers with small population risk. Unfortunately, ERM often does not yield models that are robust to adversarially crafted examples (Goodfellow et al., 2014; Kurakin et al., 2016; Moosavi-Dezfooli et al., 2016; Tramér et al., 2017b). Formally, there are efficient algorithms ("adversaries") that take an example  $x$  belonging to class  $c_{1}$  as input and find examples  $x^{\mathrm{adv}}$  such that  $x^{\mathrm{adv}}$  is very close to  $x$  but the model incorrectly classifies  $x^{\mathrm{adv}}$  as belonging to class  $c_{2} \neq c_{1}$ .

In order to reliably train models that are robust to adversarial attacks, it is necessary to augment the ERM paradigm appropriately. Instead of resorting to methods that directly focus on improving the robustness to specific attacks, our approach is to first propose a concrete guarantee that an adversarially robust model should satisfy. We then adapt our training methods towards achieving this guarantee.

The first step towards such a guarantee is to specify an threat model, i.e., a precise definition of the attacks our models should be resistant to. For each data point  $x$ , we introduce a set of allowed perturbations  $S \subseteq \mathbb{R}^d$  that formalizes the manipulative power of the adversary. In image classification, we choose  $S$  so that it captures perceptual similarity between images. For instance, the  $\ell_{\infty}$ -ball around  $x$  has recently been studied as a natural notion for adversarial perturbations (Goodfellow et al., 2014). While we focus on robustness against  $\ell_{\infty}$ -bounded attacks in this paper, we remark that more comprehensive notions of perceptual similarity are an important direction for future research.

Next, we modify the definition of population risk  $\mathbb{E}_{\mathcal{D}}[L]$  by incorporating the above adversary. Instead of computing the loss  $L$  directly on samples from the distribution  $\mathcal{D}$ , we allow the adversary to perturb the input first. This gives rise to the following saddle point problem, which is our central object of study:

$$
\min  _ {\theta} \rho (\theta), \quad \text {w h e r e} \quad \rho (\theta) = \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \max  _ {\delta \in \mathcal {S}} L (\theta , x + \delta , y) \right]. \tag {2.1}
$$

Formulations of this type (and their finite-sample counterparts) have a long history in robust optimization, going back to Wald (Wald, 1939; 1945; 1992). It turns out that this formulation is also particularly useful in our context. We will refer to the quantity  $\rho(\theta)$  as the adversarial loss of the network with parameters  $\theta$ .

First, this formulation gives us a unifying perspective that encompasses much prior work on adversarial robustness. Our perspective stems from viewing the saddle point problem as the composition of an inner maximization problem and an outer minimization problem. Both of these problems have a natural interpretation in our context. The inner maximization problem aims to find an adversarial version of a given data point  $x$  that achieves a high loss. This is precisely the problem of attacking a given neural network. On the other hand, the goal of the outer minimization problem is to find model parameters so that the adversarial loss given by the inner attack problem is minimized. This is precisely the problem of training a robust classifier using adversarial training techniques.

Second, the saddle point problem specifies a clear goal that an ideal robust classifier should achieve, as well as a quantitative measure of its robustness. In particular, when the parameters  $\theta$  yield a (nearly) vanishing risk, the corresponding model is perfectly robust to attacks specified by our threat model.

Our paper investigates the structure of this saddle point problem in the context of deep neural networks. This formulation will be the main drive of our investigations that will lead us to training techniques that produce models with high resistance to a wide range of adversarial attacks.

# 3 TOWARDS ADVERSARIALLY ROBUST NETWORKS

Current work on adversarial examples usually focuses on specific defensive mechanisms, or on attacks against such defenses. An important feature of formulation (2.1) is that attaining small adversarial loss gives a guarantee that no allowed attack will fool the network. By definition, no

adversarial perturbations are possible because the loss is small for all perturbations allowed by our threat model. This perspective allows us to reduce the task of finding truly robust models to an optimization problem. Hence, we can now focus our attention solely on obtaining a good solution to Problem (2.1).

Gradients from attacks. Since Stochastic Gradient Descent (SGD) and its variants are by far the most successful algorithms for training neural networks, we also want to apply SGD to Problem (2.1). This raises the question how we can compute gradients  $\nabla_{\theta}\rho (\theta)$  for the outer minimization problem. Since the adversarial loss function  $\rho (\theta)$  corresponds to a maximization problem, we cannot simply apply the usual backpropagation algorithm. Instead, a natural approach is to compute the gradient at the maximizer of the inner maximization problem. A priori, it is not clear that this is a valid descent direction for the saddle point problem. However, for the case of continuously differentiable functions, Danskin's theorem - a classic theorem in optimization - states that this is indeed true and gradients at maximizers of the inner problem correspond to descent directions for the saddle point problem (see Appendix C for details).

Leveraging this connection, our goal now is to find a reliable algorithm for solving the inner maximization problem, i.e., to evaluate  $\rho(\theta)$ . When instantiated for a batch of examples (instead of the expectation over the entire distribution  $\mathcal{D}$ ), finding a maximizer  $\delta \in S$  of  $\rho(\theta)$  corresponds exactly to finding an attack on the neural network. This allows us to employ known attacks as inner maximization algorithms. Prior work has proposed methods such as the Fast Gradient Sign Method (FGSM) and multiple variations of it (Goodfellow et al., 2014). FGSM is an attack for an  $\ell_{\infty}$ -bounded adversary and computes an adversarial example as

$$
x + \varepsilon \operatorname {s g n} (\nabla_ {x} L (\theta , x, y)).
$$

One can interpret this attack as a simple one-step scheme for maximizing the inner part of the saddle point formulation. A more powerful adversary is the multi-step variant  $\mathrm{FGSM}^k$ , which is essentially projected gradient descent (PGD) on the negative loss function (Kurakin et al., 2016):<sup>2</sup>

$$
x ^ {t + 1} = \mathrm {P r o j} _ {x + \mathcal {S}} \left(x ^ {t} + \alpha \mathrm {s g n} (\nabla_ {x} L (\theta , x, y))\right).
$$

Loss landscape. While PGD is a well-motivated approach for the inner maximization problem, it is not clear whether we can actually find a good solution in a reasonable amount of time. The problem is non-concave, so a priori we have no guarantees on the solution quality of PGD. One of our contributions is demonstrating that, in practice, the inner maximization problem is indeed well-behaved. In particular, we experimentally explore the structure given by the non-concave inner problem and find that its loss landscape has a surprisingly tractable structure of local maxima (see Appendix A). This structure also points towards projected gradient descent as the "ultimate" first-order adversary (see Section 5).

Despite the fact that the exact assumptions of Danskin's theorem do not hold for our problem (the function is not continuously differentiable due to ReLU activations, and we only compute approximate maximizers of the inner problem), our experiments suggest that we can still use these gradients to optimize our problem. By applying SGD using the gradient of the loss at adversarial examples, we can consistently reduce the loss of the saddle point problem during training (e.g., see Figure 1 in Section 4). These observations suggest that we reliably optimize the saddle point formulation (2.1) and thus train robust classifiers.

Model capacity. Before we proceed to our main experiment results in the next section, we briefly mention another important insight from our robust optimization perspective. Solving the problem from Equation (2.1) successfully is not sufficient to guarantee robust and accurate classification. We also require that the value of the problem (i.e., the final loss we achieve against adversarial examples) is small, which then provides guarantees for the performance of our classifier. In particular, achieving a very small value corresponds to a perfect classifier, which is robust to adversarial inputs. In Appendix B, we show experimentally that network capacity plays a crucial role in enabling robustness. In particular, training a robust classifier requires a significantly larger network than only achieving high accuracy on natural examples.

# 4 EXPERIMENTS: ADVERSARIALLY ROBUST DEEP LEARNING MODELS?

Following our understanding developed in the previous section, we can now apply our proposed approach to train robust classifiers. For both MNIST and CIFAR10, our adversary of choice will be projected gradient descent starting from a random perturbation around the natural example. As our experiments suggest (Appendix A) this algorithm is very efficient at reliably producing examples of (near) maximal loss. In a sense, it seems to correspond to a "ultimate" first order adversary. Since we are training the model for multiple epochs, we did not see any benefit in restarting PGD multiple times per batch - a new start is chosen each time the same example is encountered.

During the training procedure against the PGD adversary, we observe a steady decrease in the training loss of adversarial examples, illustrated in Figure 1. This behavior indicates that we are consistently decreasing the adversarial loss and indeed successfully solving our original optimization problem.

![](images/4b1a81cf6e6899273eb4b47b3c89df13c1dc7000ba38ad65396ecba095aac606.jpg)  
(a) MNIST

![](images/79da20b8767dd34969eacdc3d186f70ee46cb9c465a4b31970241bac29d97979.jpg)  
(b) CIFAR10  
Figure 1: Cross-entropy loss on adversarial examples during training. The plots show how the adversarial loss on training examples evolves during training the MNIST and CIFAR10 networks against a PGD adversary. The sharp drops in the CIFAR10 plot correspond to decreases in training learning rate. These plots illustrate that we can consistently reduce the value of the inner problem of the saddle point formulation (2.1), thus producing an increasingly robust classifier.

We evaluate the trained models against a range of adversaries. We illustrate our results in Table 1 for MNIST and Table 2 for CIFAR10. The adversaries we consider are:

- White-box attacks with PGD for a different number of iterations and restarts, denoted by source A.  
- White-box attacks from Carlini & Wagner (2016b). We use their suggested loss function and minimize it using PGD. This is denoted as CW, where the corresponding attack with a high confidence parameter  $(\kappa = 50)$  is denoted as  $\mathrm{CW}+$ .  
- Black-box attacks from an independently trained copy of the network, denoted A'.  
- Black-box attacks from a version of the same network trained only on natural examples, denoted  $A_{nat}$ .  
- Black-box attacks from a different convolution architecture, denoted B, described in Tramér et al. (2017a).

MNIST. We run 40 iterations of projected gradient descent as our adversary, with a step size of 0.01 (we choose to take gradient steps in the  $\ell_{\infty}$  norm, i.e. adding the sign of the gradient, since this makes the choice of the step size simpler). We train and evaluate against perturbations of size  $\varepsilon = 0.3$ . We use a network consisting of two convolutional layers with 32 and 64 filters respectively, each followed by  $2 \times 2$  max-pooling, and a fully connected layer of size 1024. When trained with natural examples, this network reaches  $99.2\%$  accuracy on the evaluation set. However, when evaluating on examples perturbed with FGSM the accuracy drops to  $6.4\%$ . Given that the resulting MNIST model is very robust, we investigated the learned parameters in order to understand how they affect adversarial robustness. The results of the investigation are presented in Appendix E.

CIFAR10. For the CIFAR10 dataset, we use the two architectures described in B (the original Resnet and its  $10 \times$  wider variant). We trained the network against a PGD adversary with  $\ell_{\infty}$  projected

<table><tr><td>Method</td><td>Steps</td><td>Restarts</td><td>Source</td><td>Accuracy</td></tr><tr><td>Natural</td><td>-</td><td>-</td><td>-</td><td>98.8%</td></tr><tr><td>FGSM</td><td>-</td><td>-</td><td>A</td><td>95.6%</td></tr><tr><td>PGD</td><td>40</td><td>1</td><td>A</td><td>93.2%</td></tr><tr><td>PGD</td><td>100</td><td>1</td><td>A</td><td>91.8%</td></tr><tr><td>PGD</td><td>40</td><td>20</td><td>A</td><td>90.4%</td></tr><tr><td>PGD</td><td>100</td><td>20</td><td>A</td><td>89.3%</td></tr><tr><td>Targeted</td><td>40</td><td>1</td><td>A</td><td>92.7%</td></tr><tr><td>CW</td><td>40</td><td>1</td><td>A</td><td>94.0%</td></tr><tr><td>CW+</td><td>40</td><td>1</td><td>A</td><td>93.9%</td></tr><tr><td>FGSM</td><td>-</td><td>-</td><td>A&#x27;</td><td>96.8%</td></tr><tr><td>PGD</td><td>40</td><td>1</td><td>A&#x27;</td><td>96.0%</td></tr><tr><td>PGD</td><td>100</td><td>20</td><td>A&#x27;</td><td>95.7%</td></tr><tr><td>CW</td><td>40</td><td>1</td><td>A&#x27;</td><td>97.0%</td></tr><tr><td>CW+</td><td>40</td><td>1</td><td>A&#x27;</td><td>96.4%</td></tr><tr><td>FGSM</td><td>-</td><td>-</td><td>B</td><td>95.4%</td></tr><tr><td>PGD</td><td>40</td><td>1</td><td>B</td><td>96.4%</td></tr><tr><td>CW+</td><td>-</td><td>-</td><td>B</td><td>95.7%</td></tr></table>

Table 1: MNIST: Performance of the adversarially trained network against different adversaries for  $\varepsilon = 0.3$ . For each model of attack we show the most successful attack with bold. The source networks used for the attack are: the network itself (A) (white-box attack), an independently initialized and trained copy of the network (A'), architecture B from Tramère et al. (2017a) (B).  

<table><tr><td>Method</td><td>Steps</td><td>Source</td><td>Accuracy</td></tr><tr><td>Natural</td><td>-</td><td>-</td><td>87.3%</td></tr><tr><td>FGSM</td><td>-</td><td>A</td><td>56.1%</td></tr><tr><td>PGD</td><td>7</td><td>A</td><td>50.0%</td></tr><tr><td>PGD</td><td>20</td><td>A</td><td>45.8%</td></tr><tr><td>CW</td><td>30</td><td>A</td><td>46.8%</td></tr><tr><td>FGSM</td><td>-</td><td>A&#x27;</td><td>67.0%</td></tr><tr><td>PGD</td><td>7</td><td>A&#x27;</td><td>64.2%</td></tr><tr><td>CW</td><td>30</td><td>A&#x27;</td><td>78.7%</td></tr><tr><td>FGSM</td><td>-</td><td>Anat</td><td>85.6%</td></tr><tr><td>PGD</td><td>7</td><td>Anat</td><td>86.0%</td></tr></table>

Table 2: CIFAR10: Performance of the adversarially trained network against different adversaries for  $\varepsilon = 8$ . For each model of attack we show the most effective attack in bold. The source networks considered for the attack are: the network itself (A) (white-box attack), an independently initialized and trained copy of the network (A'), a copy of the network trained on natural examples ( $A_{nat}$ ).

gradient descent again, this time using 7 steps of size 2, and a total  $\varepsilon = 8$ . For our hardest adversary we chose 20 steps with the same settings, since other hyperparameter choices didn't offer a significant decrease in accuracy. The results of our experiments appear in Table 2.

The adversarial robustness of our network is significant, given the power of iterative adversaries, but still far from satisfactory. We believe that further progress is possible along these lines by understanding how adversarial training works and what techniques can complement it leading to robust models.

Resistance for different values of  $\varepsilon$  and  $\ell_2$ -bounded attacks. In order to perform a broader evaluation of the adversarial robustness of our models, we run two kinds of additional experiments. On one hand, we investigate the resistance to  $\ell_{\infty}$ -bounded attacks for different values of  $\varepsilon$ . On the other hand, we examine the resistance of our model to attacks that are bounded in  $\ell_2$  as opposed to  $\ell_{\infty}$  norm. The results appear in Figure 2. We emphasize that the models we are examining

![](images/39a408ec0a22d1a358cc20977b29910445c06ae5338f3eb210339220be2a3fa6.jpg)  
(a) MNIST,  $\ell_{\infty}$  norm

![](images/b2a9d8c6a41192420b6c8c503eee6ec43239ab3ca8c19eb5d94740bf4a7323e2.jpg)  
(b) MNIST,  $\ell_2$  norm  
Figure 2: Performance of our adversarially trained networks against PGD adversaries of different strength. The MNIST and CIFAR10 networks were trained against  $\varepsilon = 0.3$  and  $\varepsilon = 8$  PGD  $\ell_{\infty}$  adversaries respectively (the training  $\varepsilon$  is denoted with a red dashed lines in the  $\ell_{\infty}$  plots). We notice that for  $\varepsilon$  less or equal to the value used during training, the performance is equal or better. For MNIST there is a sharp drop shortly after.

![](images/b8297e2b9ea98d14bfb1e8238f065a8564cf718c1a5fd99747d8ec9d6506a9c2.jpg)  
(c) CIFAR10,  $\ell_{\infty}$  norm

![](images/846e71a6cacd65ef233b294b0ab357344f37364bac8f7e1871e81d6d81b31d39.jpg)  
(d) CIFAR10,  $\ell_2$  norm

here correspond to training against  $\ell_{\infty}$ -bounded attacks with the original value of  $\varepsilon = 0.3$ , for MNIST, and  $\varepsilon = 8$  for CIFAR10. In particular, our MNIST model retains significant resistance to  $\ell_{2}$ -norm-bounded perturbations too – it has quite good accuracy in this regime even for  $\varepsilon = 4.5$ . To put this value of  $\varepsilon$  into perspective, we provide a sample of corresponding adversarial examples in Figure 12 of Appendix F. One can observe that some of the underlying perturbations are large enough that even a human could be confused.

# 5 FIRST-ORDER ADVERSARIES.

Our exploration of the loss landscape (Appendix A) shows that the local maxima found by PGD all have similar loss values, both for normally trained networks and adversarially trained networks. This concentration phenomenon suggests an intriguing view on the problem in which robustness against the PGD adversary yields robustness against all first-order adversaries, i.e., attacks that rely only on first-order information. As long as the adversary only uses gradients of the loss function with respect to the input, we conjecture that it will not find significantly better local maxima than PGD. This hypothesis is validated by the experimental evidence provided in Section 4: if we train a network to be robust against PGD adversaries, it becomes robust against a wide range of other attacks as well.

Of course, our exploration with PGD does not preclude the existence of some isolated maxima with much larger function value. However, our experiments suggest that such better local maxima are hard to find with first-order methods: even a large number of random restarts did not find function values with significantly different loss values (see Appendix A). Incorporating the computational power of the adversary into the threat model should be reminiscent of the notion of polynomially bounded adversary that is a cornerstone of modern cryptography. There, this classic threat model allows the adversary to only solve problems that require at most polynomial computation time. Here, we employ an optimization-based view on the power of the adversary as it is more suitable in the context of machine learning. After all, we have not yet developed a thorough understanding of the computational complexity of many recent machine learning problems. However, the vast majority of optimization problems in ML is solved with first-order methods, and variants of SGD are the most effective way of training deep learning models in particular. Hence we believe that the class of attacks relying on first-order information is, in some sense, universal for the current practice of deep learning.

Put together, these two ideas chart the way towards machine learning models with guaranteed robustness. If we train the network to be robust against PGD adversaries, it will be robust against a wide range of attacks that encompasses all current approaches.

In fact, this robustness guarantee would become even stronger in the context of transfer attacks, i.e., attacks in which the adversary does not have a direct access to the target network. Instead, the adversary only has less specific information such as the (rough) model architecture and the training data set. One can view this threat model as an example of "zero order" attacks, i.e., attacks in which the adversary has no direct access to the classifier and is only able to evaluate it on chosen examples without gradient feedback. Still, even for the case of zero-order attacks, the gradient of the network

can be estimated using a finite differences method, rendering first-order attacks also relevant in this context.

We discuss transferability in Appendix D. We observe that increasing network capacity and strengthening the adversary we train against (FGSM or PGD training, rather than natural training) improves resistance against transfer attacks. Also, as expected, the resistance of our best models to such attacks tends to be significantly larger than to the (strongest) first order attacks.

# 6 RELATED WORK

Due to the growing body of work on adversarial examples in the context of deep learning networks (Gu & Rigazio, 2014; Fawzi et al., 2015; Torkamani, 2016; Papernot et al., 2016b; Carlini & Wagner, 2016a; Tramér et al., 2017b; Goodfellow et al., 2014; Kurakin et al., 2016), we focus only on the most related papers here. Before we compare our contributions, we remark that robust optimization has been studied outside deep learning for multiple decades. We refer the reader to Ben-Tal et al (2009) for an overview of this field.

Recent work on adversarial training on ImageNet also observed that the model capacity is important for adversarial training Kurakin et al. (2016). However, their work was focused on FGSM attacks, since they report the iterative attacks are too expensive computationally and don't provide any significant benefits. In contrast to that, we discover that for the datasets we considered training against iterative adversaries does result in a model that is robust against such adversaries.

In Huang et al. (2015) and Shaham et al. (2015) a version of the min-max optimization problem is also considered for adversarial training. Both of these works, however, consider very weak adversaries/methods for solving the maximization problem, mainly relying on linearizing the loss and performing a single step, similar to FGSM. These adversaries do not capture the full range of possible attacks and thus training only against them leaves the resulting classifier vulnerable to more powerful, iterative attacks. Additionally, while the experiments in Shaham et al. (2015) produce promising results, they are only evaluated against FGSM. However, FGSM-only evaluations are not fully reliable. One evidence for that is that Shaham et al. (2015) reports  $70\%$  accuracy for  $\varepsilon = 0.7$ , but any adversary that is allowed to perturb each pixel by more than 0.5 can construct a uniformly gray image, thus fooling any classifier.

A more recent paper Tramér et al. (2017b) also explores the transferability phenomenon. This exploration focuses mostly on the region around natural examples where the loss is (close to) linear. When large perturbations are allowed, this region does not give a complete picture of the adversarial landscape. This is confirmed by our experiments, as well as pointed out by Tramér et al. (2017a).

Another recent paper Tramèr et al. (2017a), considers adversarial training using black-box attacks from similar networks in order to increase the robustness of the network against such adversaries. However, this is not an effective defense against the white-box setting we consider, since a PGD adversary can reliably produce adversarial examples for such networks.

# 7 CONCLUSION

Our findings provide evidence that deep neural networks can be made resistant to adversarial attacks. As our theory and experiments indicate, we can design reliable adversarial training methods. One of the key insights behind this is the unexpectedly regular structure of the underlying optimization task: even though the relevant problem corresponds to the maximization of a highly non-concave function with many distinct local maxima, their values are highly concentrated. Overall, our findings give us hope that adversarially robust deep learning models may be within current reach.

For the MNIST dataset, our networks are very robust, achieving high accuracy for a wide range of powerful adversaries and large perturbations. Our experiments on CIFAR10 have not reached the same level of performance yet. However, our results already show that our techniques lead to significant increase in the robustness of the network. We believe that further exploring this direction will lead to adversarially robust networks for this dataset.

# REFERENCES

Tensor flow models repository. https://github.com/tensorflow/models/tree/master/resnet, 2017.  
Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust optimization. Princeton University Press, 2009.  
Nicholas Carlini and David Wagner. Defensive distillation is not robust to adversarial examples. arXiv preprint arXiv:1607.04311, 2016a.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. arXiv preprint arXiv:1608.04644, 2016b.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. arXiv preprint arXiv:1705.07263, 2017.  
Nilesh Dalvi, Pedro Domingos, Mausam, Sumit Sanghai, and Deepak Verma. Adversarial classification. In International Conference on Knowledge Discovery and Data Mining (KDD), 2004.  
Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Analysis of classifiers' robustness to adversarial perturbations. arXiv preprint arXiv:1502.02590, 2015.  
Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Shixiang Gu and Luca Rigazio. Towards deep neural network architectures robust to adversarial examples. arXiv preprint arXiv:1412.5068, 2014.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 2016.  
Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song. Adversarial example defenses: Ensembles of weak defenses are not strong. arXiv preprint arXiv:1706.04701, 2017.  
Ruitong Huang, Bing Xu, Dale Schuurmans, and Csaba Szepesvári. Learning with a strong adversary. arXiv preprint arXiv:1511.03034, 2015.  
Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial machine learning at scale. arXiv preprint arXiv:1611.01236, 2016.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: A simple and accurate method to fool deep neural networks. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pp. 2574-2582, 2016.  
Anh Mai Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pp. 427-436, 2015.  
Nicolas Papernot and Patrick D. McDaniel. On the effectiveness of defensive distillation. arXiv preprint arXiv:1607.05113, 2016.  
Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation as a defense to adversarial perturbations against deep neural networks. In Security and Privacy (SP), 2016 IEEE Symposium on, pp. 582-597. IEEE, 2016a.  
Nicolas Papernot, Patrick D. McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In IEEE European Symposium on Security and Privacy, EuroS&P 2016, Saarbrücken, Germany, March 21-24, 2016, pp. 372-387, 2016b.  
Andras Rozsa, Manuel Gunther, and Terrance E. Boult. Towards robust deep neural networks with BANG. arXiv preprint arXiv:1612.00138, 2016.

Uri Shaham, Yutaro Yamada, and Sahand Negahban. Understanding adversarial training: Increasing local stability of neural nets through robust optimization. arXiv preprint arXiv:1511.05432, 2015.  
Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K. Reiter. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, Vienna, Austria, October 24-28, 2016, pp. 1528-1540, 2016.  
Jure Sokolic, Raja Giryes, Guillermo Sapiro, and Miguel RD Rodrigues. Robust large margin deep neural networks. arXiv preprint arXiv:1605.08254, 2016.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
MohamadAli Torkamani. Robust Large Margin Approaches for Machine Learning in Adversarial Settings. PhD thesis, University of Oregon, 2016.  
Florian Tramèr, Alexey Kurakin, Nicolas Papernot, Dan Boneh, and Patrick D. McDaniel. Ensemble adversarial training: Attacks and defenses. arXiv preprint arXiv:1705.07204, 2017a.  
Florian Tramèr, Nicolas Papernot, Ian J. Goodfellow, Dan Boneh, and Patrick D. McDaniel. The space of transferable adversarial examples. arXiv preprint arXiv:1704.03453, 2017b. URL http://arxiv.org/abs/1704.03453.  
Abraham Wald. Contributions to the theory of statistical estimation and testing hypotheses. The Annals of Mathematical Statistics, 10(4):299-326, 1939.  
Abraham Wald. Statistical decision functions which minimize the maximum risk. Annals of Mathematics, pp. 265-280, 1945.  
Abraham Wald. Statistical decision functions. In *Breakthroughs in Statistics*, pp. 342-357. Springer, 1992.  
Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. arXiv preprint arXiv:1704.01155, 2017.
