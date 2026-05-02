# ENSEMBLE ADVERSARIAL TRAINING: ATTACKS AND DEFENSES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Adversarial examples are perturbed inputs designed to fool machine learning models. Adversarial training injects such examples into training data to increase robustness. To scale this technique to large datasets, perturbations are crafted using fast single-step methods that maximize a linear approximation of the model's loss. We show that this form of adversarial training converges to a degenerate global minimum, wherein small curvature artifacts near the data points obfuscate a linear approximation of the loss. The model thus learns to generate weak perturbations, rather than defend against strong ones. As a result, we find that adversarial training remains vulnerable to black-box attacks, where we transfer perturbations computed on undefended models, as well as to a powerful novel single-step attack that escapes the non-smooth vicinity of the input data via a small random step.

We further introduce Ensemble Adversarial Training, a technique that augments training data with perturbations transferred from other models. On ImageNet, Ensemble Adversarial Training yields models with strong robustness to black-box attacks. In particular, our most robust model won the first round of the NIPS 2017 competition on Defenses against Adversarial Attacks (Kurakin et al., 2017c).

# 1 INTRODUCTION

Machine learning (ML) models are often vulnerable to adversarial examples, maliciously perturbed inputs designed to mislead a model at test time (Biggio et al., 2013; Szegedy et al., 2013; Goodfellow et al., 2014b; Papernot et al., 2016a). Furthermore, Szegedy et al. (2013) showed that these inputs transfer across models: the same adversarial example is often misclassified by different models, thus enabling simple black-box attacks on deployed models (Papernot et al., 2017; Liu et al., 2017).

Adversarial training (Szegedy et al., 2013) increases robustness by augmenting training data with adversarial examples. Madry et al. (2017) showed that adversarially trained models can be made robust to white-box attacks (i.e., with knowledge of the model parameters) if the perturbations computed during training closely maximize the model's loss. However, prior attempts at scaling this approach to ImageNet-scale tasks Deng et al. (2009) have proven unsuccessful (Kurakin et al., 2017b).

It is thus natural to ask whether it is possible, at scale, to achieve robustness against the class of black-box adversaries Towards this goal, Kurakin et al. (2017b) adversarially trained an Inception v3 model (Szegedy et al., 2016b) on ImageNet using a "single-step" attack based on a linearization of the model's loss (Goodfellow et al., 2014b). Their trained model is robust to single-step perturbations but remains vulnerable to more costly "multi-step" attacks. Yet, Kurakin et al. (2017b) found that these attacks fail to reliably transfer between models, and thus concluded that the robustness of their model should extend to black-box adversaries. Surprisingly, we show that this is not the case.

We demonstrate, formally and empirically, that adversarial training with single-step methods admits a degenerate global minimum, wherein the model's loss can not be reliably approximated by a linear function. Specifically, we find that the model's decision surface exhibits sharp curvature near the data points, thus degrading attacks based on a single gradient computation. In addition to the model of Kurakin et al. (2017b), we reveal similar overfitting in an adversarially trained Inception ResNet v2 model (Szegedy et al., 2016a), and a variety of models trained on MNIST (LeCun et al., 1998).

We harness this result in two ways. First, we show that adversarially trained models using single-step methods remain vulnerable to simple attacks. For black-box adversaries, we find that perturbations

crafted on an undefended model often transfer to an adversarially trained one. We also introduce a simple yet powerful single-step attack that applies a small random perturbation—to escape the nonsmooth vicinity of the data point—before linearizing the model's loss. While seemingly weaker than the Fast Gradient Sign Method of Goodfellow et al. (2014b), our attack significantly outperforms it for a same perturbation norm, for models trained with or without adversarial training.

Second, we propose Ensemble Adversarial Training, a training methodology that incorporates perturbed inputs transferred from other pre-trained models. Our approach decouples adversarial example generation from the parameters of the trained model, and increases the diversity of perturbations seen during training. We train Inception v3 and Inception ResNet v2 models on ImageNet that exhibit increased robustness to adversarial examples transferred from other holdout models, using various single-step and multi-step attacks (Goodfellow et al., 2014b; Carlini & Wagner, 2017a; Kurakin et al., 2017a; Madry et al., 2017). We also show that our methods globally reduce the dimensionality of the space of adversarial examples (Tramér et al., 2017). Our Inception ResNet v2 model won the first round of the NIPS 2017 competition on Defenses Against Adversarial Attacks (Kurakin et al., 2017c), where it was evaluated on other competitors' attacks in a black-box setting.<sup>1</sup>

# 2 RELATED WORK

Various defensive techniques against adversarial examples in deep neural networks have been proposed (Gu & Rigazio, 2014; Luo et al., 2015; Papernot et al., 2016c; Nayebi & Ganguli, 2017; Cisse et al., 2017) and many remain vulnerable to adaptive attackers (Carlini & Wagner, 2017a;b; Baluja & Fischer, 2017). Adversarial training (Szegedy et al., 2013; Goodfellow et al., 2014b; Kurakin et al., 2017b; Madry et al., 2017) appears to hold the greatest promise for learning robust models.

Madry et al. (2017) show that adversarial training on MNIST yields models that are robust to white-box attacks, if the adversarial examples used in training closely maximize the model's loss. As we argue in Appendix B, the MNIST dataset is peculiar in that there exists a simple "closed-form" denoising procedure (namely feature binarization) which leads to similarly robust models without adversarial training. This may explain why robustness to white-box attacks is hard to scale to tasks such as ImageNet (Kurakin et al., 2017b). We believe that the existence of a simple robust baseline for MNIST can be useful for understanding some limitations of adversarial training techniques.

Szegedy et al. (2013) found that adversarial examples transfer between models, thus enabling black-box attacks on deployed models. Papernot et al. (2017) showed that black-box attacks could succeed with no access to training data, by exploiting the target model's predictions to extract (Tramér et al., 2016) a surrogate model. Some prior works have hinted that adversially trained models may remain vulnerable to black-box attacks: Goodfellow et al. (2014b) found that an adversarial maxout network on MNIST has slightly higher error on transferred examples than on white-box examples. Papernot et al. (2017) further showed that a model trained on small perturbations can be evaded by transferring perturbations of larger magnitude. Our finding that adversarial training degrades the accuracy of linear approximations of the model's loss is as an instance of a gradient-masking phenomenon (Papernot et al., 2016b), which affects other defensive techniques (Papernot et al., 2016c; Carlini & Wagner, 2017a; Nayebi & Ganguli, 2017; Brendel & Bethge, 2017).

# 3 THE ADVERSARIAL TRAINING FRAMEWORK

We consider a classification task with data  $x \in [0,1]^d$  and labels  $y_{\mathrm{true}} \in \mathbb{Z}_k$  sampled from a distribution  $\mathcal{D}$ . We identify a model with an hypothesis  $h$  from a space  $\mathcal{H}$ . On input  $x$ , the model outputs class scores  $h(x) \in \mathbb{R}^k$ . The loss function used to train the model, e.g., cross-entropy, is  $L(h(x),y)$ .

# 3.1 THREAT MODEL

For some target model  $h \in \mathcal{H}$  and inputs  $(x, y_{\mathrm{true}})$  the adversary's goal is to find an adversarial example  $x^{\mathrm{adv}}$  such that  $x^{\mathrm{adv}}$  and  $x$  are "close" yet the model misclassifies  $x^{\mathrm{adv}}$ . We consider the well-studied class of  $\ell_{\infty}$  bounded adversaries (Goodfellow et al., 2014b; Madry et al., 2017) that, given

some budget  $\epsilon$ , output examples  $x^{\mathrm{adv}}$  where  $\| x^{\mathrm{adv}} - x \|_{\infty} \leq \epsilon$ . As we comment in Appendix B.1,  $\ell_{\infty}$  robustness is of course not an end-goal for secure ML. We use this standard model to showcase limitations of prior adversarial training methods, and evaluate our proposed improvements.

We distinguish between white-box adversaries that have access to the target model's parameters (i.e.,  $h$ ), and black-box adversaries that only interact with a model's prediction interface. Formal definitions for these adversaries are in Appendix A. Although security against white-box attacks is the stronger notion (and the one we ideally want ML models to achieve), black-box security is a reasonable and more tractable goal for deployed ML models.

# 3.2 ADVERSARIAL TRAINING

Following Madry et al. (2017), we consider an adversarial variant of standard Empirical Risk Minimization (ERM), where our aim is to minimize the risk over adversarial examples:

$$
h ^ {*} = \underset {h \in \mathcal {H}} {\arg \min } \quad \underset {(x, y _ {\text {t r u e}}) \sim \mathcal {D}} {\mathbb {E}} \left[ \max  _ {\| x ^ {\mathrm {a d v}} - x \| _ {\infty} \leq \epsilon} L (h \left(x ^ {\mathrm {a d v}}\right), y _ {\text {t r u e}}) \right]. \tag {1}
$$

Madry et al. (2017) argue that adversarial training has a natural interpretation in this context, where a given attack (see below) is used to approximate solutions to the inner maximization problem, and the outer minimization problem corresponds to training over these examples. Note that the original formulation of adversarial training (Szegedy et al., 2013; Goodfellow et al., 2014b), which we use in our experiments, trains on both the "clean" examples  $x$  and adversarial examples  $x^{\mathrm{adv}}$ .

We consider three algorithms to generate adversarial examples with bounded  $\ell_{\infty}$  norm. The first two are single-step (i.e., they require a single gradient computation); the third is iterative—it computes multiple gradient updates. We enforce  $x^{\mathrm{adv}} \in [0,1]^d$  by clipping all components of  $x^{\mathrm{adv}}$ .

Fast Gradient Sign Method (FGSM). This method (Goodfellow et al., 2014b) linearizes the inner maximization problem in (1):

$$
x ^ {\mathrm {a d v}} = x + \varepsilon \cdot \operatorname {s i g n} \left(\nabla_ {x} L (h (x), y _ {\text {t r u e}})\right). \tag {2}
$$

Single-Step Least-Likely Class Method (Step-LL). This variant of FGSM introduced by Kurakin et al. (2017a;b) targets the least-likely class,  $y_{\mathrm{LL}} = \arg \min \{h(x)\}$ :

$$
x ^ {\mathrm {a d v}} = x - \varepsilon \cdot \operatorname {s i g n} \left(\nabla_ {x} L (h (x), y _ {\mathrm {L L}})\right). \tag {3}
$$

Although this attack only indirectly tackles the inner maximization in (1), Kurakin et al. (2017b) find it to be the most effective for adversarial training on ImageNet.

Iterative Attack (I-FGSM or Iter-LL). This method iteratively applies the FGSM or Step-LL  $k$  times with step-size  $\alpha \geq \epsilon /k$  and projects each step onto the  $\ell_{\infty}$  ball of norm  $\epsilon$  around  $x$ . It uses projected gradient descent to solve the maximization in (1). For fixed  $\epsilon$ , iterative attacks induce higher error rates than single-step attacks, but transfer at lower rates (Kurakin et al., 2017a;b).

# 3.3 A DEGENERATE GLOBAL MINIMUM FOR SINGLE-STEP ADVERSARIAL TRAINING

When performing adversarial training with a single-step attack (e.g., the FGSM or Step-LL methods above), we approximate Equation (1) by replacing the solution to the inner maximization problem with the output of the single-step attack. For model families  $\mathcal{H}$  with high expressive power, this alternative optimization problem admits at least two substantially different global minima  $h^*$ :

Minimum 1: On average, for an input  $x$  from  $\mathcal{D}$ , there is no input  $x^{\mathrm{adv}}$  close to  $x$  (in  $\ell_{\infty}$  norm) that induces a high loss. In other words,  $h^{*}$  is robust to all  $\ell_{\infty}$  bounded perturbations.  
Minimum 2: The minimizer  $h^*$  is a model for which the approximation method underlying the attack (i.e., linearization in our case) poorly fits the model's loss function. As a result, the attack when applied to  $h^*$  produces samples  $x^{\mathrm{adv}}$  that are far from optimal.

Note that this second "degenerate" minimum can be more subtle than a simple case of overfitting to samples produced from single-step attacks. Indeed, we show in Section 4.1 that single-step attacks applied to adversarially trained models create "adversarial" examples that are easy to classify even for undefended models. Thus, adversarial training does not simply learn to resist the particular attack used during training, but actually to make that attack perform worse overall.

![](images/b056e4a65e0ca47225ecc2c9443ad0318d57d249270b9e4f7216a0296798b371.jpg)  
(a) Loss of model v3adv.

![](images/5c1c515aa77b228d8b766b8553c7da60fe52f2eecc2c8dc4b0bb9806bb0de92d.jpg)  
(b) Zoom in for small  $\epsilon_1, \epsilon_2$ .  
Figure 1: Gradient masking in single-step adversarial training. We plot the loss of model  $\mathrm{v3}_{\mathrm{adv}}$  on points  $x^{*} = x + \epsilon_{1}\cdot g + \epsilon_{2}\cdot g^{\perp}$ , where  $g$  is the signed gradient and  $g^{\perp}$  is an orthogonal adversarial direction. Plot (b) is a zoom of (a) near  $x$ . The gradient poorly approximates the global loss.

# 3.4 ENSEMBLE ADVERSARIAL TRAINING

The degenerate minimum described in Section 3.3 is attainable because the learned model's parameters influence the quality of both the minimization and maximization in (1). One solution is to use a stronger adversarial example generation process, at a high performance cost (Madry et al., 2017). Alternatively, Baluja & Fischer (2017) suggest training an adversarial generator model as in the GAN framework (Goodfellow et al., 2014a). The power of this generator is likely to require careful tuning, to avoid similar degenerate minima (where the generator or classifier overpowers the other).

We propose a conceptually simpler approach to decouple the generation of adversarial examples from the model being trained, while simultaneously drawing an explicit connection with robustness to black-box adversaries. Our method, which we call Ensemble Adversarial Training, augments a model's training data with adversarial examples crafted on other static pre-trained models. Intuitively, as adversarial examples transfer between models, perturbations crafted on an external model are good approximations for the maximization problem in (1). Moreover, the learned model can not influence the "strength" of these adversarial examples. As a result, minimizing the training loss implies increased robustness to black-box attacks from some set of models. We introduce the following (informal) conjecture, which we will show to be supported by our experimental results (see 4.2):

Robustness to attacks transfers: That is, a model that learns to be robust to attacks transferred from some models  $\{h_1,\ldots ,h_k\}$  is also more robust to black-box attacks from other models.

# 4 EXPERIMENTS

We show the existence of a degenerate minimum, as described in Section 3.3, for the adversarially trained Inception v3 model of Kurakin et al. (2017b). Their model (denoted  $\mathrm{v}3_{\mathrm{adv}}$ ) was trained on a Step-LL attack with  $\epsilon \leq 16 / 256$ . We also adversarially train an Inception ResNet v2 model (Szegedy et al., 2016a) using the same setup. We denote this model by  $\mathrm{IRv}2_{\mathrm{adv}}$ . We refer the reader to (Kurakin et al., 2017b) for details on the adversarial training procedure.

We first measure the approximation-ratio of the Step-LL attack for the inner maximization in (1). As we do not know the true maximum, we lower-bound it using an iterative attack. For 1,000 random test points, we find that for a standard Inception v3 model, step-LL gets within  $19\%$  of the optimum loss on average. This attack is thus a good candidate for adversarial training. Yet, for the  $\mathrm{v3}_{\mathrm{adv}}$  model, the approximation ratio drops to  $7\%$ , confirming that the learned model is less amenable to linearization. We obtain similar results for Inception ResNet v2 models. The ratio is  $17\%$  for a standard model, and  $8\%$  for IRv2<sup>adv</sup>. Similarly, we look at the cosine similarity between the perturbations given by a single-step and multi-step attack. The more linear the model, the more similar we expect both perturbations to be. The average similarity drops from 0.13 for Inception v3 to 0.02 for  $\mathrm{v3}_{\mathrm{adv}}$ . This effect is not due to the decision surface of  $\mathrm{v3}_{\mathrm{adv}}$  being "too flat" near the data points: the average gradient norm is larger for  $\mathrm{v3}_{\mathrm{adv}}$  (0.17) than for the standard v3 model (0.10).

Table 1: Error rates (in %) of adversarial examples transferred between models. We use Step-LL with  $\epsilon = 16 / 256$  for 10,000 random test inputs. Diagonal elements represent a white-box attack. The best attack for each target appears in bold. Similar results for MNIST models appear in Table 7.  

<table><tr><td rowspan="2">Target</td><td colspan="5">Source</td><td rowspan="2">Target</td><td colspan="5">Source</td></tr><tr><td>v4</td><td>v3</td><td>v3adv</td><td>IRv2</td><td>IRv2adv</td><td>v4</td><td>v3</td><td>v3adv</td><td>IRv2</td><td>IRv2adv</td></tr><tr><td>v4</td><td>60.2</td><td>39.2</td><td>31.1</td><td>36.6</td><td>30.9</td><td>v4</td><td>31.0</td><td>14.9</td><td>10.2</td><td>13.6</td><td>9.9</td></tr><tr><td>v3</td><td>43.8</td><td>69.6</td><td>36.4</td><td>42.1</td><td>35.1</td><td>v3</td><td>18.7</td><td>42.7</td><td>13.0</td><td>17.8</td><td>12.8</td></tr><tr><td>v3adv</td><td>36.3</td><td>35.6</td><td>26.6</td><td>35.2</td><td>35.9</td><td>v3adv</td><td>13.6</td><td>13.5</td><td>9.0</td><td>13.0</td><td>14.5</td></tr><tr><td>IRv2</td><td>38.0</td><td>38.0</td><td>30.8</td><td>50.7</td><td>31.9</td><td>IRv2</td><td>14.1</td><td>14.8</td><td>9.9</td><td>24.0</td><td>10.6</td></tr><tr><td>IRv2adv</td><td>31.0</td><td>30.3</td><td>25.7</td><td>30.6</td><td>21.4</td><td>IRv2adv</td><td>10.3</td><td>10.5</td><td>7.7</td><td>10.4</td><td>5.8</td></tr><tr><td colspan="6">Top 1</td><td colspan="6">Top 5</td></tr></table>

We visualize this "gradient-masking" effect (Papernot et al., 2016b) by plotting the loss of  $\mathrm{v3}_{\mathrm{adv}}$  on examples  $x^{*} = x + \epsilon_{1}\cdot g + \epsilon_{2}\cdot g^{\perp}$ , where  $g$  is the signed gradient of model  $\mathrm{v3}_{\mathrm{adv}}$  and  $g^{\perp}$  is a signed vector orthogonal to  $g$ . Looking forward to Section 4.1, we actually chose  $g^{\perp}$  to be the signed gradient of another Inception model, from which adversarial examples transfer to  $\mathrm{v3}_{\mathrm{adv}}$ . Figure 1 shows that the loss is highly curved in the vicinity of the data point  $x$ , and that the gradient poorly reflects the global loss landscape. Similar plots for additional data points are in Figure 4.

We show similar results for adversarially trained MNIST models in Appendix B.2. On this task, input dropout (Srivastava et al., 2014) mitigates adversarial training's overfitting problem, in some cases. Presumably, the random input mask diversifies the perturbations seen during training (dropout at intermediate layers does not mitigate the overfitting effect). Mishkin et al. (2017) find that input dropout significantly degrades accuracy on ImageNet, so we did not include it in our experiments.

# 4.1 ATTACKS AGAINST ADVERSARIALLY TRAINED NETWORKS

Kurakin et al. (2017b) found their adversarially trained model to be robust to various single-step attacks. They conclude that this robustness should translate to attacks transferred from other models. As we have shown, the robustness to single-step attacks is actually misleading, as the model has learned to degrade the information contained in the model's gradient. As a consequence, we find that the  $\mathrm{v3}_{\mathrm{adv}}$  model is substantially more vulnerable to single-step attacks than Kurakin et al. (2017b) predicted, both in a white-box and black-box setting. The same holds for the IRv2<sub>adv</sub> model.

In addition to the v3<sub>adv</sub> and IRv2<sub>adv</sub> models, we consider standard Inception v3, Inception v4 and Inception ResNet v2 models. These models are available in the TensorFlow-Slim library (Abadi et al., 2015). We describe similar results for a variety of models trained on MNIST in Appendix B.2.

Black-box attacks. Table 1 shows error rates for single-step attacks transferred between models. We compute perturbations on one model (the source) and transfer them to all others (the targets). When the source and target are the same, the attack is white-box. Adversarial training greatly increases robustness to white-box single-step attacks, but incurs a higher error rate in a black-box setting. Thus, the robustness gain observed when evaluating defended models in isolation is misleading. Given the ubiquity of this pitfall among proposed defenses against adversarial examples Carlini & Wagner (2017a); Brendel & Bethge (2017); Papernot et al. (2016b), we advise researchers to always consider both white-box and black-box adversaries when evaluating defensive strategies.

Attacks crafted on adversarial models are found to be weaker even against undefended models (i.e., when using  $\mathrm{v3}_{\mathrm{adv}}$  or  $\mathrm{IRv2}_{\mathrm{adv}}$  as source, the attack transfers with lower probability). This confirms our intuition from Section 3.3: adversarial training does not just overfit to perturbations that affect standard models, but actively degrades the linear approximation underlying the single-step attack.

A new randomized single-step attack. The loss function visualization in Figure 1 shows that sharp curvature artifacts localized near the data points can mask the true direction of steepest ascent. We thus suggest to prepend single-step attacks by a small random step, in order to "escape" the non-smooth vicinity of the data point before linearizing the model's loss. Our new attack, called R+FGSM (alternatively, R+Step-LL), is defined as follows, for parameters  $\epsilon$  and  $\alpha$  (where  $\alpha < \epsilon$ ):

$$
x ^ {\text {a d v}} = x ^ {\prime} + (\varepsilon - \alpha) \cdot \operatorname {s i g n} \left(\nabla_ {x ^ {\prime}} J \left(x ^ {\prime}, y _ {\text {t r u e}}\right)\right), \quad \text {w h e r e} \quad x ^ {\prime} = x + \alpha \cdot \operatorname {s i g n} \left(\mathcal {N} \left(\mathbf {0} ^ {d}, \mathbf {I} ^ {d}\right)\right). \tag {4}
$$

Table 2: Error rates (in %) for Step-LL and R+Step-LL on ImageNet. We use  $\epsilon = 16 / 256$ ,  $\alpha = \epsilon /2$  on 10,000 random test set inputs. Results for R+FGSM on MNIST models are in Table 7.  

<table><tr><td></td><td>v4</td><td>v3</td><td>v3adv</td><td>IRv2</td><td>IRv2adv</td><td>v4</td><td>v3</td><td>v3adv</td><td>IRv2</td><td>IRv2adv</td></tr><tr><td>Step-LL</td><td>60.2</td><td>69.6</td><td>26.6</td><td>50.7</td><td>21.4</td><td>31.0</td><td>42.7</td><td>9.0</td><td>24.0</td><td>5.8</td></tr><tr><td>R+Step-LL</td><td>70.5</td><td>80.0</td><td>64.8</td><td>56.3</td><td>37.5</td><td>42.8</td><td>57.1</td><td>37.1</td><td>29.3</td><td>15.0</td></tr><tr><td></td><td colspan="5">Top 1</td><td colspan="5">Top 5</td></tr></table>

Note that the attack requires a single gradient computation. The R+FGSM is a computationally efficient alternative to iterative methods that have high success rates in a white-box setting. Our attack can be seen as a single-step variant of the general PGD method from (Madry et al., 2017).

Table 2 compares error rates for the Step-LL and R+Step-LL methods (with  $\epsilon = 16 / 256$  and  $\alpha = \epsilon /2$ ). The extra random step yields a stronger attack for all models, even those without adversarial training. This suggests that a model's loss function is generally less smooth near the data points.

We find that the addition of this random step hinders transferability (see Table 9). We also tried adversarial training using R+FGSM on MNIST, using a similar approach as (Madry et al., 2017). We adversarially train a CNN (model A in Table 5) for 100 epochs, and attain  $>90.0\%$  accuracy on R+FGSM samples. However, training on R+FGSM provides only little robustness to iterative attacks. For the PGD attack of (Madry et al., 2017) with 20 steps, the model attains  $18.0\%$  accuracy.

# 4.2 ENSEMBLE ADVERSARIAL TRAINING

We now evaluate our Ensemble Adversarial Training strategy described in Section 3.4. We recall our intuition: by augmenting training data with adversarial examples crafted from static pre-trained models, we decouple the generation of adversarial examples from the model being trained, so as to avoid the degenerate minimum described in Section 3.3. Moreover, our hope is that robustness to attacks transferred from some fixed set of models will generalize to other black-box adversaries.

Table 3: Models used for Ensemble Adversarial Training on ImageNet. The ResNets (He et al., 2016) use either 50 or 101 layers. IncRes stands for Inception ResNet (Szegedy et al., 2016a).  

<table><tr><td>Trained Model</td><td>Pre-trained Models</td><td>Holdout Models</td></tr><tr><td>Inception v3 (v3adv-ens3)</td><td>Inception v3, ResNet v2 (50)</td><td>Inception v4</td></tr><tr><td>Inception v3 (v3adv-ens4)</td><td>Inception v3, ResNet v2 (50), IncRes v2</td><td>ResNet v1 (50)</td></tr><tr><td>IncRes v2 (IRv2adv-ens)</td><td>Inception v3, IncRes v2</td><td>ResNet v2 (101)</td></tr></table>

We train Inception v3 and Inception ResNet v2 models (Szegedy et al., 2016a) on ImageNet, using the pre-trained models shown in Table 3. In each training batch, we rotate the source of adversarial examples between the currently trained model and one of the pre-trained models. We select the source model at random in each batch, to diversify examples across epochs. The pre-trained models' gradients can be precomputed for the full training set. The per-batch cost of Ensemble Adversarial Training is thus lower than that of standard adversarial training: using our method with  $n - 1$  pre-trained models, only every  $n^{\mathrm{th}}$  batch requires a forward-backward pass to compute adversarial gradients. We use synchronous distributed training on 50 machines, with minibatches of size 16 (we did not pre-compute gradients, and thus lower the batch size to fit all models in memory). Half of the examples in a minibatch are replaced by Step-LL examples. As in (Kurakin et al., 2017b), we use RMSProp with a learning rate of 0.045, decayed by a factor of 0.94 every two epochs.

To evaluate how robustness to black-box attacks generalizes across models (our conjecture in Section 3.4), we transfer various attacks crafted on three different holdout models (see Table 3). We use the Step-LL, R+Step-LL, FGSM, I-FGSM and the PGD attack from Madry et al. (2017) using the hinge-loss function from Carlini & Wagner (2017a). Our results are in Table 4. For each model, we report the worst-case error rate over all black-box attacks transferred from each of the holdout models (15 attacks in total). Additional results on MNIST are in Table 8.

Convergence speed. Convergence of Ensemble Adversarial Training is slower than for standard adversarial training, a result of training on "hard" adversarial examples and lowering the batch size. Kurakin et al. (2017b) report that after 187 epochs (150k iterations with minibatches of size 32),

Table 4: Error rates (in %) for Ensemble Adversarial Training on ImageNet. Error rates on clean data are computed over the full test set. For 10,000 random test set inputs, and  $\epsilon = 16 / 256$ , we report error rates on white-box Step-LL and the worst-case error over a series of black-box attacks (Step-LL,  $R +$  Step-LL, FGSM, I-FGSM, PGD) transferred from the holdout models in Table 3. For both architectures, we mark methods tied for best in bold (based on  $95\%$  confidence).  

<table><tr><td rowspan="2">Model</td><td colspan="3">Top 1</td><td colspan="3">Top 5</td></tr><tr><td>Clean</td><td>Step-LL</td><td>Max. Black-Box</td><td>Clean</td><td>Step-LL</td><td>Max. Black-Box</td></tr><tr><td>v3</td><td>22.0</td><td>69.6</td><td>51.2</td><td>6.1</td><td>42.7</td><td>24.5</td></tr><tr><td>\( v_{3_{\text{adv}}} \)</td><td>22.0</td><td>26.6</td><td>40.8</td><td>6.1</td><td>9.0</td><td>17.4</td></tr><tr><td>\( v_{3_{\text{adv-ens3}}} \)</td><td>23.6</td><td>30.0</td><td>34.0</td><td>7.6</td><td>10.1</td><td>11.2</td></tr><tr><td>\( v_{3_{\text{adv-ens4}}} \)</td><td>24.2</td><td>43.3</td><td>33.4</td><td>7.8</td><td>19.4</td><td>10.7</td></tr><tr><td>IRv2</td><td>19.6</td><td>50.7</td><td>44.4</td><td>4.8</td><td>24.0</td><td>17.8</td></tr><tr><td>\( IRv2_{\text{adv}} \)</td><td>19.8</td><td>21.4</td><td>34.5</td><td>4.9</td><td>5.8</td><td>11.7</td></tr><tr><td>\( IRv2_{\text{adv-ens}} \)</td><td>20.2</td><td>26.0</td><td>27.0</td><td>5.1</td><td>7.6</td><td>7.9</td></tr></table>

the  $\mathrm{v3}_{\mathrm{adv}}$  model achieves 78% accuracy. Ensemble Adversarial Training for models  $\mathrm{v3}_{\mathrm{adv - ens3}}$  and  $\mathrm{v3}_{\mathrm{adv - ens4}}$  converges after 280 epochs (450k iterations with minibatches of size 16). The Inception ResNet v2 model is trained for 175 epochs, where a baseline model converges at around 160 epochs.

White-box attacks. For both architectures, the models trained with Ensemble Adversarial Training are slightly less accurate on clean data, compared to standard adversarial training. Our models are also more vulnerable to white-box single-step attacks, as they were only partially trained on such perturbations. Note that for  $\mathrm{v3}_{\mathrm{adv - ens4}}$ , the proportion of white-box Step-LL samples seen during training is  $1/4$  (instead of  $1/3$  for model  $\mathrm{v3}_{\mathrm{adv - ens3}}$ ). The negative impact on the robustness to white-box attacks is large, for only a minor gain in robustness to transferred samples.

Ensemble Adversarial Training is not robust to white-box Iter-LL and R+Step-LL samples: the error rates are similar to those for the v3<sub>adv</sub> model, and omitted for brevity (see Kurakin et al. (2017b) for Iter-LL attacks and Table 2 for R+Step-LL attacks). Kurakin et al. (2017b) conjecture that larger models are needed to attain robustness to such attacks. Yet, against black-box adversaries, these attacks are only a concern insofar as they reliably transfer between models.

Black-box attacks. Ensemble Adversarial Training significantly boosts robustness to all attacks transferred from the holdout models, thus supporting our conjecture from Section 3.4 that robustness transfers across models. For the  $\mathrm{IRv2}_{\mathrm{adv - ens}}$  model, the accuracy loss (compared to IRv2's accuracy on clean data) is  $7.4\%$  (top 1) and  $3.1\%$  (top 5). We find that the strongest attacks in our test suite (i.e., with highest transfer rates) are single-step attacks. Black-box R+Step-LL or iterative attacks are less effective, as they do not transfer with high probability (see (Kurakin et al., 2017b) and Table 9). We also tried the I-FGSM crafted on an ensemble of all three holdout models, as in (Liu et al., 2017) but we also found this attack to be inferior to the transferred single-step attacks.

Our results have little variance with respect to the attack parameters (e.g., smaller  $\epsilon$ ) or to the use of other holdout models for black-box attacks (e.g., we obtain similar results by attacking the v3<sup>adv-ens3</sup> and v3<sup>adv-ens4</sup> models with the IRv2 model). We also find that v3<sup>adv-ens3</sup> is not vulnerable to perturbations transferred from v3<sup>adv-ens4</sup>. We obtain similar results on MNIST (see Appendix B.2), thus demonstrating the applicability of our approach to different datasets and model architectures.

Our Inception ResNet v2 model was included as a baseline defense in the NIPS 2017 competition on Adversarial Examples (Kurakin et al., 2017c). The model was evaluated on other users' black-box attacks, and finished  $I^{st}$  among 70 submissions in the first development round. After the first round, we released our model publicly, which enabled other users to launch white-box attacks against it. Nevertheless, a majority of the teams' final submissions were ensembles that included our model.

Decreasing gradient masking. Ensemble Adversarial Training decreases the magnitude of the gradient masking effect described previously. For the v3<sub>adv-ens3</sub> and v3<sub>adv-ens4</sub> models, we find that the loss incurred on a Step-LL attack gets within respectively 13% and 18% of the optimum loss (we recall that for models v3 and v3<sub>adv</sub>, the approximation ratio was respectively 19% and 7%). Similarly, for the IRv2<sub>adv-ens</sub> model, the ratio improves from 8% (for IRv2<sub>adv</sub>) to 14%. As expected, not solely training on a white-box single-step attack decreases the gradient masking effect.

![](images/7ad9015a3539c0a4d21e32b0fe01ac2313b4cf545dc4288929aa8037f2600191.jpg)  
Figure 2: The dimensionality of the adversarial cone. For 500 correctly classified points  $x$ , and for  $\epsilon \in \{4,10,16\}$ , we plot the probability that we find at least  $k$  orthogonal vectors  $r_i$  such that  $\| r_i \|_{\infty} = \epsilon$  and  $x + r_i$  is misclassified. For  $\epsilon \geq 10$ , model v3<sup>adv</sup> shows a bimodal phenomenon: most points  $x$  either have 0 adversarial directions or more than 90.

![](images/23abb514840e9ef5c8635374877febf39590f569481b4c93c12a64c10f7d3ffa.jpg)

![](images/2b0b23d20bcebf5a6c34a769c1675514b1e860f8972424e3e63f05c5688b0c0f.jpg)

Finally, we revisit the "Gradient-Aligned Adversarial Subspace" (GAAS) method of Tramér et al. (2017). Their method estimates the size of the space of adversarial examples in the vicinity of a point, by finding a set of orthogonal perturbations of norm  $\epsilon$  that are all adversarial. We note that adversarial perturbations do not technically form a "subspace" (e.g., the 0 vector is not adversarial). Rather, they may form a "cone", the dimension of which varies as we increase  $\epsilon$ . By linearizing the loss function, estimating the dimensionality of this cone reduces to finding vectors  $r_i$  that are strongly aligned with the model's gradient  $g = \nabla_x L(h(x), y_{\mathrm{true}})$ . Tramér et al. (2017) give a method that finds  $k$  orthogonal vectors  $r_i$  that satisfy  $g^\top r_i \geq \epsilon \cdot \| g \|_2 \cdot \frac{1}{\sqrt{k}}$  (this bound is tight). We extend this result to the  $\ell_\infty$  norm, an open question in (Tramér et al., 2017). In Section D, we give a randomized combinatorial construction (Colbourn, 2010), that finds  $k$  orthogonal vectors  $r_i$  satisfying  $\| r_i \|_\infty = \epsilon$  and  $\mathbb{E}\left[g^\top r_i\right] \geq \epsilon \cdot \| g \|_1 \cdot \frac{1}{\sqrt{k}}$ . We show that this result is tight as well.

For models v3, v3<sup>adv</sup> and v3<sup>adv-ens3</sup>, we select 500 correctly classified test points. For each  $x$ , we search for a maximal number of orthogonal adversarial perturbations  $r_i$  with  $\| r_i \|_{\infty} = \epsilon$ . We limit our search to  $k \leq 100$  directions per point. The results are in Figure 2. For  $\epsilon \in \{4, 10, 16\}$ , we plot the proportion of points that have at least  $k$  orthogonal adversarial perturbations. For a fixed  $\epsilon$ , the value of  $k$  can be interpreted as the dimension of a "slice" of the cone of adversarial examples near a data point. For the standard Inception v3 model, we find over 50 orthogonal adversarial directions for  $30\%$  of the points. The v3<sup>adv</sup> model shows a curious bimodal phenomenon for  $\epsilon \geq 10$ : for most points  $(\approx 80\%)$ , we find no adversarial direction aligned with the gradient, which is consistent with the gradient masking effect. Yet, for most of the remaining points, the adversarial space is very high-dimensional ( $k \geq 90$ ). Ensemble Adversarial Training yields a more robust model, with only a small fraction of points near a large adversarial space.

# 5 CONCLUSION AND FUTURE WORK

Previous work on adversarial training at scale has produced encouraging results, showing strong robustness to (single-step) adversarial examples (Goodfellow et al., 2014b; Kurakin et al., 2017b). Yet, these results are misleading, as the adversarially trained models remain vulnerable to simple black-box and white-box attacks. Our results, generic with respect to the application domain, suggest that adversarial training can be improved by decoupling the generation of adversarial examples from the model being trained. Our experiments with Ensemble Adversarial Training support our conjecture from Section 3.4, that robustness to attacks from some models transfers to other models.

We did not consider black-box adversaries that attack a model via other means than by transferring examples from a local model (see Appendix A for a formal definition). For instance, generative techniques (Baluja & Fischer, 2017) might provide an avenue for stronger attacks. Moreover, adaptive adversaries (also defined in Appendix A) could try to exploit queries to the target model's prediction function in their attack, as demonstrated in (Papernot et al., 2017). If queries to the target model yield full prediction confidences, an adversary could estimate the target's gradient at a given point (e.g., using finite-differences) and fool the target with our R+FGSM attack. If queries only return the predicted label, the attack does not apply. Exploring the impact of these classes of black-box attacks and evaluating their scalability to complex tasks is an interesting avenue for future work.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL https://www.tensorflow.org/. Software available from tensorflow.org.  
Shumeet Baluja and Ian Fischer. Adversarial transformation networks: Learning to generate adversarial examples. arXiv preprint arXiv:1703.09387, 2017.  
Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić, Pavel Laskov, Giorgio Giacinto, and Fabio Roli. Evasion attacks against machine learning at test time. In ECML-KDD, pp. 387-402. Springer, 2013.  
Wieland Brendel and Matthias Bethge. Comment on" biologically inspired protection of deep networks from adversarial attacks". arXiv preprint arXiv:1704.01547, 2017.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy, 2017a.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. arXiv preprint arXiv:1705.07263, 2017b.  
Moustapha Cisse, Bojanowski Piotr, Grave Edouard, Dauphin Yann, and Usunier Nicolas. Parseval networks: Improving robustness to adversarial examples. arXiv preprint arXiv:1704.08847, 2017.  
Charles J Colbourn. CRC handbook of combinatorial designs. CRC press, 2010.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014a.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014b.  
Shixiang Gu and Luca Rigazio. Towards deep neural network architectures robust to adversarial examples. arXiv preprint arXiv:1412.5068, 2014.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In ICLR, 2017a.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale. In ICLR, 2017b.  
Alexey Kurakin, Ian J Goodfellow, and Samy Bengio. Nips 2017: Defense against adversarial attack, 2017c. URL https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delving into transferable adversarial examples and black-box attacks. In ICLR, 2017.

Yan Luo, Xavier Boix, Gemma Roig, Tomaso Poggio, and Qi Zhao. Foveation-based mechanisms alleviate adversarial examples. arXiv preprint arXiv:1511.06292, 2015.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Dmytro Mishkin, Nikolay Sergievskiy, and Jiri Matas. Systematic evaluation of convolution neural network advances on theImagenet. Computer Vision and Image Understanding, 2017.  
Aran Nayebi and Surya Ganguli. Biologically inspired protection of deep networks from adversarial attacks. arXiv preprint arXiv:1703.09202, 2017.  
Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In Security and Privacy (EuroS&P), 2016 IEEE European Symposium on, pp. 372-387. IEEE, 2016a.  
Nicolas Papernot, Patrick McDaniel, Arunesh Sinha, and Michael Wellman. Towards the science of security and privacy in machine learning. arXiv preprint arXiv:1611.03814, 2016b.  
Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation as a defense to adversarial perturbations against deep neural networks. In *Security and Privacy (SP)*, 2016 IEEE Symposium on, pp. 582-597. IEEE, 2016c.  
Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. Practical black-box attacks against machine learning. In Asia Conference on Computer and Communications Security (ASIACCS), pp. 506-519. ACM, 2017.  
Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of machine learning research, 15(1):1929-1958, 2014.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alex Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv preprint arXiv:1602.07261, 2016a.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, pp. 2818-2826, 2016b.  
Florian Tramér, Fan Zhang, Ari Juels, Michael K Reiter, and Thomas Ristenpart. Stealing machine learning models via prediction apis. In Usenix Security, 2016.  
Florian Tramér, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. The space of transferable adversarial examples. arXiv preprint arXiv:1704.03453, 2017.
