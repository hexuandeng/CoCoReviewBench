# EVALUATING ROBUSTNESS OF NEURAL NETWORKS WITH MIXED INTEGER PROGRAMMING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Neural networks trained only to optimize for training accuracy can often be fooled by adversarial examples — slightly perturbed inputs misclassified with high confidence. Verification of networks enables us to gauge their vulnerability to such adversarial examples. We formulate verification of piecewise-linear neural networks as a mixed integer program. On a representative task of finding minimum adversarial distortions, our verifier is two to three orders of magnitude quicker than the state-of-the-art. We achieve this computational speedup via tight formulations for non-linearities, as well as a novel presolve algorithm that makes full use of all information available. The computational speedup allows us to verify properties on convolutional and residual networks with over 100,000 RLUs — several orders of magnitude more than networks previously verified by any complete verifier. In particular, we determine for the first time the exact adversarial accuracy of an MNIST classifier to perturbations with bounded  $l_{\infty}$  norm  $\epsilon = 0.1$ : for this classifier, we find an adversarial example for  $4.38\%$  of samples, and a certificate of robustness to norm-bounded perturbations for the remainder. Across all robust training procedures and network architectures considered, and for both the MNIST and CIFAR-10 datasets, we are able to certify more samples than the state-of-the-art and find more adversarial examples than a strong first-order attack.

# 1 INTRODUCTION

Neural networks trained only to optimize for training accuracy have been shown to be vulnerable to adversarial examples: perturbed inputs that are very similar to some regular input but for which the output is radically different (Szegedy et al., 2014). There is now a large body of work proposing defense methods to produce classifiers that are more robust to adversarial examples. However, as long as a defense is evaluated only via heuristic attacks (such as the Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2015) or Carlini & Wagner (2017b)'s attack (CW)), we have no guarantee that the defense actually increases the robustness of the classifier produced. Defense methods thought to be successful when published have often been later found to be vulnerable to a new class of attacks. For instance, multiple defense methods are defeated in Carlini & Wagner (2017a) by constructing defense-specific loss functions and in Athalye et al. (2018) by overcoming obfuscated gradients.

Fortunately, we can evaluate robustness to adversarial examples in a principled fashion. One option is to determine (for each test input) the minimum distance to the closest adversarial example, which we call the minimum adversarial distortion (Carlini et al., 2017). Alternatively, we can determine the adversarial test accuracy (Bastani et al., 2016), which is the proportion of the test set for which no perturbation in some bounded class causes a misclassification. An increase in the mean minimum adversarial distortion or in the adversarial test accuracy indicates an improvement in robustness.<sup>1</sup>

We present an efficient implementation of a mixed-integer linear programming (MILP) verifier for properties of piecewise-linear feed-forward neural networks. Our tight formulation for nonlinearities and our novel presolve algorithm combine to minimize the number of binary variables in the MILP problem and dramatically improve its numerical conditioning. Optimizations in our MILP implementation improve performance by several orders of magnitude when compared to a naive

MILP implementation, and we are two to three orders of magnitude faster than the state-of-the-art Satisfiability Modulo Theories (SMT) based verifier, Reluplex (Katz et al., 2017)

We make the following key contributions:

- We demonstrate that, despite considering the full combinatorial nature of the network, our verifier can succeed on larger neural networks, including those with convolutional and residual layers, when evaluating the robustness of these networks to bounded perturbations.  
- We identify why we can succeed on larger neural networks with hundreds of thousands of units. First, a large fraction of the ReLUs can be shown to be either always active or always inactive over the bounded input domain. Second, since the predicted label is determined by the unit in the final layer with the maximum activation, proving that a unit never has the maximum activation over all bounded perturbations eliminates it from consideration. We fully exploit both phenomena, reducing the overall number of non-linearities considered.  
- We determine for the first time the exact adversarial accuracy for MNIST classifiers to perturbations with bounded  $l_{\infty}$  norm  $\epsilon$ . We are also able to certify more samples than the state-of-the-art and find more adversarial examples across MNIST and CIFAR-10 classifiers with different architectures trained with a variety of robust training procedures.

Our code will be made available after review.

# 2 RELATED WORK

Our work relates most closely to other work on verification of piecewise-linear neural networks; Bunel et al. (2017) provides a good overview of the field. We categorize verification procedures as complete or incomplete. To understand the difference between these two types of procedures, we consider the example of evaluating adversarial accuracy.

As in Kolter & Wong (2017), we call the exact set of all final-layer activations that can be achieved by applying a bounded perturbation to the input the adversarial polytope. Incomplete verifiers reason over an outer approximation of the adversarial polytope. As a result, when using incomplete verifiers, the answer to some queries about the adversarial polytope may not be decidable. In particular, incomplete verifiers can only certify robustness for a fraction of robust input; the status for the remaining input is undetermined. In contrast, complete verifiers reason over the exact adversarial polytope. Given sufficient time, a complete verifier can provide a definite answer to any query about the adversarial polytope. In the context of adversarial accuracy, complete verifiers will obtain a valid adversarial example or a certificate of robustness for every input. When a time limit is set, complete verifiers behave like incomplete verifiers, and resolve only a fraction of queries. However, complete verifiers do allow users to answer a larger fraction of queries by extending the set time limit.

Incomplete verifiers for evaluating network robustness employ a range of techniques, including duality (Dvijotham et al., 2018; Kolter & Wong, 2017; Raghunathan et al., 2018), layer-by-layer approximations of the adversarial polytope (Xiang et al., 2018), discretizing the search space (Huang et al., 2017), abstract interpretation (Gehr et al., 2018), bounding the local Lipschitz constant (Weng et al., 2018), or bounding the activation of the ReLU with linear functions (Weng et al., 2018).

Complete verifiers typically employ either MILP solvers as we do (Cheng et al., 2017; Dutta et al., 2018; Fischetti & Jo, 2018; Lomuscio & Maganti, 2017) or SMT solvers (Carlini et al., 2017; Ehlers, 2017; Katz et al., 2017; Scheibler et al., 2015). Our approach improves upon existing MILP-based approaches with a tighter formulation for non-linearities and a novel presolve algorithm that makes full use of all information available, leading to solve times several orders of magnitude faster than a naively implemented MILP-based approach. When comparing our approach to the state-of-the-art SMT-based approach (Reluplex) on the task of finding minimum adversarial distortions, we find that our verifier is two to three orders of magnitude faster. Crucially, these improvements in performance allow our verifier to verify a network with over 100,000 units — several orders of magnitude larger than the largest MNIST classifier previously verified with a complete verifier.

A complementary line of research to verification is in robust training procedures that train networks designed to be robust to bounded perturbations. Robust training attempts to minimize the "worst-case loss" for each example — that is, the maximum loss over all bounded perturbations of that example

(Kolter & Wong, 2017). Since calculating the exact worst-case loss can be computationally costly, robust training procedures typically instead minimize an estimate of the worst-case loss: either a lower bound as in the case of adversarial training (Goodfellow et al., 2015), or an upper bound as for certified training approaches (Hein & Andriushchenko, 2017; Kolter & Wong, 2017; Raghunathan et al., 2018). Complete verifiers such as ours can augment robust training procedures by resolving the status of input for which heuristic attacks cannot find an adversarial example but incomplete verifiers cannot guarantee robustness, enabling more accurate comparisons between different training procedures.

# 3 BACKGROUND AND NOTATION

We denote a neural network by a function  $f(\cdot ;\theta):\mathbb{R}^m\to \mathbb{R}^n$  parameterized by a (fixed) vector of weights  $\theta$ . For a classifier, the output layer has a neuron for each target class.

Verification as solving an MILP. The general problem of verification is to determine whether some property  $P$  on the output of a neural network holds for all input in a bounded input domain  $\mathcal{C} \subseteq \mathbb{R}^m$ . For the verification problem to be expressible as solving an MILP,  $P$  must be expressible as the conjunction or disjunction of linear properties  $P_{i,j}$  over some set of polyhedra  $\mathcal{C}_i$ , where  $\mathcal{C} = \cup \mathcal{C}_i$ .

In addition,  $f(\cdot)$  must be composed of piecewise-linear layers. This is not a particularly restrictive requirement: piecewise-linear layers include layers that are linear transformations (such as fully-connected, convolution, and average-pooling layers) and layers that use piecewise-linear functions (such as ReLU or maximum-pooling layers). We provide details on how to express these piecewise-linear functions in Section 4.1. The "shortcut connections" used in architectures such as ResNet (He et al., 2016) are also linear, and batch normalization (Ioffe & Szegedy, 2015) or dropout (Srivastava et al., 2014) are linear transformations at evaluation time (Bunel et al., 2017).

# 4 FORMULATING ROBUSTNESS EVALUATION OF CLASSIFIERS AS AN MILP

Evaluating Adversarial Accuracy. Let  $\mathcal{G}(x)$  denote the region in the input domain corresponding to all allowable perturbations of a particular input  $x$ . In general, perturbed inputs must also remain in the domain of valid inputs  $\mathcal{X}_{valid}$ . For example, for normalized images with pixel values ranging from 0 to 1,  $\mathcal{X}_{valid} = [0,1]^m$ . As in Madry et al. (2018), we say that a neural network is robust to perturbations on  $x$  if the predicted probability of the true label  $\lambda(x)$  exceeds that of every other label for all perturbations:

$$
\forall x ^ {\prime} \in (\mathcal {G} (x) \cap \mathcal {X} _ {\text {v a l i d}}): \operatorname {a r g m a x} _ {i} \left(f _ {i} \left(x ^ {\prime}\right)\right) = \lambda (x) \tag {1}
$$

Equivalently, the network is robust to perturbations on  $x$  if and only if Equation 2 is infeasible for  $x'$ .

$$
\left(x ^ {\prime} \in (\mathcal {G} (x) \cap \mathcal {X} _ {\text {v a l i d}})\right) \wedge \left(f _ {\lambda (x)} \left(x ^ {\prime}\right) <   \max  _ {\mu \in [ 1, n ] \backslash \{\lambda (x) \}} f _ {\mu} \left(x ^ {\prime}\right)\right) \tag {2}
$$

where  $f_{i}(\cdot)$  is the  $i^{\text{th}}$  output of the network. For conciseness, we call  $x$  robust with respect to the network if  $f(\cdot)$  is robust to perturbations on  $x$ . If  $x$  is not robust, we call any  $x'$  satisfying the constraints a valid adversarial example to  $x$ . The adversarial accuracy of a network is the fraction of the test set that is robust; the adversarial error is simply the complement of the adversarial accuracy.

As long as  $\mathcal{G}(x) \cap \mathcal{X}_{\text{valid}}$  can be expressed as the union of a set of polyhedra, the feasibility problem can be expressed as an MILP. The four robust training procedures we consider (Kolter & Wong, 2017; Wong et al., 2018; Madry et al., 2018; Raghunathan et al., 2018) are designed to be robust to perturbations with bounded  $l_{\infty}$  norm, and the  $l_{\infty}$ -ball of radius  $\epsilon$  around each input  $x$  can be succinctly represented by the set of linear constraints  $\mathcal{G}(x) = \{x' \mid \forall i: -\epsilon \leq (x - x')_i \leq \epsilon\}$ .

Evaluating Mean Minimum Adversarial Distortion. Let  $d(\cdot, \cdot)$  denote a distance metric that measures the perceptual similarity between two input images. The minimum adversarial distortion under  $d$  for input  $x$  with true label  $\lambda(x)$  corresponds to the solution to the optimization:

$$
\min  _ {x ^ {\prime}} d \left(x ^ {\prime}, x\right) \tag {3}
$$

$$
\text {s u b j e c t} \quad \operatorname {a r g m a x} _ {i} \left(f _ {i} \left(x ^ {\prime}\right)\right) \neq \lambda (x) \tag {4}
$$

$$
x ^ {\prime} \in \mathcal {X} _ {\text {v a l i d}} \tag {5}
$$

We can target the attack to generate an adversarial example that is classified in one of a set of target labels  $T$  by replacing Equation 4 with  $\operatorname{argmax}_i(f_i(x')) \in T$ .

The most prevalent distance metrics in the literature for generating adversarial examples are the  $l_{1}$  (Carlini & Wagner, 2017b; Chen et al., 2018),  $l_{2}$  (Szegedy et al., 2014), and  $l_{\infty}$  (Goodfellow et al., 2015; Papernot et al., 2016) norms. All three can be expressed in the objective without adding any additional integer variables to the model (Boyd & Vandenberghe, 2004); details are in Appendix A.3.

# 4.1 FORMULATING PIECEWISE-LINEAR FUNCTIONS IN THE CLASSIFIER

Tight formulations of the rectifier and maximum functions are critical to good performance of the MILP solver; we thus present these formulations in detail with accompanying proofs.2

Formulating ReLU Let  $y = \max(x, 0)$ , and  $l \leq x \leq u$ . There are three possibilities for the phase of the ReLU. If  $u \leq 0$ , we have  $y \equiv 0$ . We say that such a unit is stably inactive. Similarly, if  $l \geq 0$ , we have  $y \equiv x$ . We say that such a unit is stably active. Otherwise, the unit is unstable. For unstable units, we introduce an indicator decision variable  $a = \mathbb{1}_{x \geq 0}$ . As we prove in Appendix A.1,  $y = \max(x, 0)$  is equivalent to the set of linear and integer constraints in Equation 6.

$$
(y \leq x - l (1 - a)) \wedge (y \geq x) \wedge (y \leq u \cdot a) \wedge (y \geq 0) \wedge (a \in \{0, 1 \}) \tag {6}
$$

Formulating the Maximum Function Let  $y = \max (x_{1},x_{2},\ldots ,x_{m})$ , and  $l_{i}\leq x_{i}\leq u_{i}$

Proposition 1. Let  $l_{max} \triangleq \max(l_1, l_2, \ldots, l_m)$ . We can eliminate from consideration all  $x_i$  where  $u_i \leq l_{max}$ , since we know that  $y \geq l_{max} \geq u_i \geq x_i$ .

We introduce an indicator decision variable  $a_{i}$  for each of our input variables, where  $a_{i} = 1 \Rightarrow y = x_{i}$ . Furthermore, we define  $u_{\max, -i} \triangleq \max_{j \neq i}(u_j)$ . As we prove in Appendix A.2, the constraint  $y = \max(x_{1}, x_{2}, \ldots, x_{m})$  is equivalent to the set of linear and integer constraints in Equation 7.

$$
\bigwedge_ {i = 1} ^ {m} \left(\left(y \leq x _ {i} + (1 - a _ {i}) \left(u _ {\max , - i} - l _ {i}\right)\right) \wedge \left(y \geq x _ {i}\right)\right) \wedge \left(\sum_ {i = 1} ^ {m} a _ {i} = 1\right) \wedge \left(a _ {i} \in \{0, 1 \}\right) \tag {7}
$$

# 4.2 PROGRESSIVE BOUNDS TIGHTENING

We previously assumed that we had some element-wise bounds on the inputs to non-linearities. In practice, we have to carry out a presolve step to determine these bounds. Determining tight bounds is critical for problem tractability: tight bounds strengthen the problem formulation and thus improve solve times (Vielma, 2015). For instance, if we can prove that the phase of a ReLU is stable, we can avoid introducing a binary variable. More generally, loose bounds on input to some unit will propagate downstream, leading to units in later layers having looser bounds.

We used two procedures to determine bounds: INTERVAL ARITHMETIC (IA), also used in Cheng et al. (2017); Dutta et al. (2018), and the slower but tighter LINEAR PROGRAMMING (LP) approach. Implementation details are in Appendix B.

Since faster procedures achieve efficiency by compromising on tightness of bounds, we face a tradeoff between higher build times (to determine tighter bounds to inputs to non-linearities), and higher solve times (to solve the main MILP problem in Equation 2 or Equation 3-5). While a degree of compromise is inevitable, our knowledge of the non-linearities used in our network allows us to reduce average build times without affecting the strength of the problem formulation.

The key observation is that, for piecewise-linear non-linearities, there are thresholds beyond which further refining a bound will not improve the problem formulation. With this in mind, we adopt a progressive bounds tightening approach: we begin by determining coarse bounds using fast procedures and only spend time refining bounds using procedures with higher computational complexity if doing so could provide additional information to improve the problem formulation.<sup>3</sup> Pseudocode demonstrating how to efficiently determine bounds for the tightest possible formulations for the ReLU and maximum function is provided below and in Appendix C respectively.

GETBOUNDsFORReLU(x,fs)

1  $\triangleright$ $fs$  are the procedures to determine bounds, sorted in increasing computational complexity.  
2  $l_{best} = -\infty ;u_{best} = \infty$   initialize best known upper and lower bounds on  $x$  
3 for  $f$  in  $fs$ :  $\triangleright$  carrying out progressive bounds tightening  
4  $\mathbf{d}\mathbf{o}u = f(x,boundType = upper);u_{best} = \min (u_{best},u)$  
5 if  $u_{best} \leq 0$  return ( $l_{best}, u_{best}$ )  $\triangleright$  Early return:  $x \leq u_{best} \leq 0$ ; thus  $\max(x, 0) \equiv 0$ .  
6  $l = f(x, \text{boundType} = \text{lower}); l_{\text{best}} = \max(l_{\text{best}}, l)$  
7 if  $l_{best} \geq 0$  return ( $l_{best}, u_{best}$ )  $\triangleright$  Early return:  $x \geq l_{best} \geq 0$ ; thus  $\max(x, 0) \equiv x$  
8 return  $(l_{best},u_{best})\triangleright x$  could be either positive or negative.

The process of progressive bounds tightening is naturally extensible to more procedures. Kolter & Wong (2017); Wong et al. (2018); Dvijotham et al. (2018); Weng et al. (2018) each discuss procedures to determine bounds with computational complexity and tightness intermediate between IA and LP. Using one of these procedures in addition to IA and LP has the potential to further reduce build times.

# 5 EXPERIMENTS

Dataset. All experiments are carried out on classifiers for the MNIST dataset of handwritten digits or the CIFAR-10 dataset of color images.

Architectures. We conduct experiments on a range of feed-forward networks. In all cases, ReLUs follow each layer except the output layer.  $\mathbf{MLP - }m\times [n]$  refers to a multilayer perceptron with  $m$  hidden layers and  $n$  units per hidden layer. We further abbreviate  $\mathrm{MLP - 1}\times [500]$  and  $\mathrm{MLP - 2}\times [200]$  as  $\mathbf{MLP}_{\mathrm{A}}$  and  $\mathbf{MLP}_{\mathrm{B}}$  respectively. CNN refers to the ConvNet architecture used for the robust MNIST classifier in Kolter & Wong (2017). The network has two convolutional layers (stride length 2) with 16 and 32 filters respectively (size  $4\times 4$  in both layers), followed by a fully-connected layer with 100 units. RES refers to the ResNet architecture used in Wong et al. (2018), with 9 convolutional layers in four blocks, followed by two fully-connected layers with 4096 and 1000 units respectively.

Training Methods. We conduct experiments on networks trained with a regular loss function and networks trained to be robust. Networks trained to be robust are identified by a prefix corresponding to the method used to approximate the worst-case loss:  $\mathbf{LP_d}^4$  when the dual of a linear program is used, as in Kolter & Wong (2017);  $\mathbf{SDP_d}$  when the dual of a semidefinite relaxation is used, as in Raghunathan et al. (2018); and Adv when adversarial examples generated via Projected Gradient Descent (PGD) are used, as in Madry et al. (2018). Full details on each network are in Appendix D.1.

Experimental Setup. We run experiments on a modest 8 CPUs@2.20 GHz with 8GB of RAM. Appendix D.2 provides additional details about the computational environment. Maximum build effort is LP. Unless otherwise noted, we report a timeout if solve time for some input exceeds 1200s.

# 5.1 PERFORMANCE COMPARISONS

# 5.1.1 COMPARISONS TO OTHER MILP-BASED COMPLETE VERIFIERS

Our MILP approach implements three key optimizations: we use progressive tightening, make use of the information provided by the restricted input domain  $\mathcal{G}(x)$ , and use asymmetric bounds in the ReLU formulation in Equation 6. None of the four other MILP-based complete verifiers implement progressive tightening or use the restricted input domain, and only Fischetti & Jo (2018) uses asymmetric bounds. Since none of the four verifiers have publicly available code, we use ablation tests to provide an idea of the difference in performance between our verifier and these existing ones.

When removing progressive tightening, we directly use LP rather than doing IA first. When removing using restricted input domain, we determine bounds under the assumption that our perturbed input could be anywhere in the full input domain  $\mathcal{X}_{\text{valid}}$ , imposing the constraint  $x' \in \mathcal{G}(x)$  only after all bounds are determined. Finally, when removing using asymmetric bounds, we replace  $l$  and  $u$  in Equation 6 with  $-M$  and  $M$  respectively, where  $M \triangleq \max(-l, u)$ , as is done in Cheng et al. (2017); Dutta et al. (2018); Lomuscio & Maganti (2017). We carry out experiments on an MNIST classifier, and results from performing these experiments are reported in Table 1.

Table 1: Results of ablation testing on our verifier, where each test removes a single optimization. The task was to determine the adversarial accuracy of the MNIST classifier  $\mathrm{LP_d}$ -CNN to perturbations with  $l_{\infty}$  norm-bound  $\epsilon = 0.1$ . Build time refers to time used to determine bounds, while solve time refers to time used to solve the main MILP problem in Equation 2 once all bounds have been determined. During solve time, we solve a linear program for each of the nodes explored in the MILP search tree. We exclude the initial build time required (3593s) to determine reusable bounds.  

<table><tr><td rowspan="2">Optimization Removed</td><td colspan="3">Mean Time / s</td><td colspan="2">Nodes Explored</td><td rowspan="2">Fraction Timed Out</td></tr><tr><td>Build</td><td>Solve</td><td>Total</td><td>Mean</td><td>Median</td></tr><tr><td>(Control)</td><td>3.44</td><td>0.08</td><td>3.52</td><td>2.05</td><td>0</td><td>0</td></tr><tr><td>Progressive tightening</td><td>7.66</td><td>0.11</td><td>7.77</td><td>2.05</td><td>0</td><td>0</td></tr><tr><td>Using restricted input domain†</td><td>1.49</td><td>56.47</td><td>57.96</td><td>1343.21</td><td>67</td><td>0.0047</td></tr><tr><td>Using asymmetric bounds</td><td>4465.11</td><td>133.03</td><td>4598.15</td><td>1498.35</td><td>113</td><td>0.0300</td></tr></table>

The ablation tests demonstrate that each optimization is critical to the performance of our verifier. In terms of performance comparisons, we expect our verifier to have a runtime several orders of magnitude faster than any of the three verifiers not using asymmetric bounds. While Fischetti & Jo (2018) do use asymmetric bounds, they do not use information from the restricted input domain; we thus expect our verifier to have a runtime at least an order of magnitude faster than theirs.

# 5.1.2 COMPARISONS TO OTHER COMPLETE AND INCOMPLETE VERIFIERS

We also compared our verifier to other verifiers on the task of finding minimum targeted adversarial distortions for MNIST test samples. Verifiers included for comparison are 1) Reluplex (Katz et al., 2017), a complete verifier also able to find the true minimum distortion; and 2) LP<sup>5</sup>, Fast-Lip, Fast-Lin (Weng et al., 2018), and LP-full (Kolter & Wong, 2017), incomplete verifiers that provide a certified lower bound on the minimum distortion.

Verification Times, vis-à-vis the state-of-the-art SMT-based complete verifier Reluplex. Figure 1 presents average verification times per sample. All solves for our method were run to completion. On the  $l_{\infty}$  norm, we improve on the speed of Reluplex by two to three orders of magnitude.

![](images/925619fc0f03231497076dda49e71db9ea56b702513c0d48b852e31202453d25.jpg)  
Figure 1: Average times for determining bounds on / exact values of the minimum targeted adversarial distortion for MNIST test samples. We improve on the speed of the state-of-the-art complete verifier Reluplex by two to three orders of magnitude. Results for methods other than ours are from Weng et al. (2018); results for Reluplex were only available in Weng et al. (2018) for the  $l_{\infty}$  norm.

Minimum Targeted Adversarial Distortions, vis-à-vis incomplete verifiers. Figure 2 compares lower bounds from the incomplete verifiers to the exact value we obtain. The gap between the best lower bound and the true minimum adversarial distortion is significant even on these small networks. This corroborates the observation in Raghunathan et al. (2018) that incomplete verifiers provide weak bounds if the network they are applied to is not optimized for that verifier. For example, under the  $l_{\infty}$  norm, the best certified lower bound is less than half of the true minimum distortion. In context: a network robust to perturbations with  $l_{\infty}$  norm-bound  $\epsilon = 0.1$  would only be verifiable to  $\epsilon = 0.05$ .

![](images/4907a0bc4a3f65e5c0b6b7b63f72588cb6095732fd28804cb069857b9e7370e3.jpg)  
Figure 2: Average of bounds on / exact values of the minimum targeted adversarial distortion for MNIST test samples. The gap between the true minimum adversarial distortion and the best lower bound is significant and increases for deeper networks.

![](images/d145254c047b2764836ecf2ebaab4ed1d31d5ecb2d850b7146025a753326711f.jpg)

![](images/6fd97c2517c118f8bd83972ada2e84e68cd282e2f4c745c19ea28bc6871863b4.jpg)

# 5.2 DETERMINING ADVERSARIAL ACCURACY OF MNIST AND CIFAR-10 CLASSIFIERS

We use our verifier to determine the adversarial accuracy of classifiers trained by a range of robust training procedures on the MNIST and CIFAR-10 datasets. Table 2 presents the test error and estimates of the adversarial error for these classifiers. For MNIST, we verified a range of networks trained to be robust to attacks with bounded  $l_{\infty}$  norm  $\epsilon = 0.1$ , as well as networks trained to be robust to larger attacks of  $\epsilon = 0.2, 0.3$  and 0.4. Lower bounds on the adversarial error are proven by providing adversarial examples for input that is not robust. We compare the number of samples for which we successfully find adversarial examples to the number for PGD, a strong first-order attack. Upper bounds on the adversarial error are proven by providing certificates of robustness for input that is robust. We compare our upper bounds to the previous state-of-the-art for each network.

Table 2: Adversarial accuracy of MNIST and CIFAR-10 classifiers to perturbations with  $l_{\infty}$  norm-bound  $\epsilon$ . In every case, we improve on both 1) the lower bound on the adversarial error, found by PGD, and 2) the previous state-of-the-art (SOA) for the upper bound, generated by the following methods: [1] Kolter & Wong (2017), [2] Dvijotham et al. (2018), [3] Raghunathan et al. (2018). For classifiers marked with a  $\sqrt{}$ , we have a guarantee of robustness or a valid adversarial example for every test sample. Gaps between our bounds correspond to cases where the solver reached the time limit for some samples. Additional solve statistics on nodes explored are in Appendix E.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Network</td><td rowspan="2">ε</td><td rowspan="2">Test Error</td><td colspan="5">Certified Bounds on Adversarial Error</td><td rowspan="2">Mean Time /s</td></tr><tr><td>Lower PGD</td><td>Bound Ours</td><td>Upper SOA</td><td>Bound Ours</td><td>No Gap?</td></tr><tr><td rowspan="7">MNIST</td><td>LPd-CNN</td><td>0.1</td><td>1.89%</td><td>4.11%</td><td>4.38%</td><td>5.82% [1]</td><td>4.38%</td><td>✓</td><td>3.52</td></tr><tr><td>Adv-CNN</td><td>0.1</td><td>0.96%</td><td>4.10%</td><td>4.21%</td><td>—</td><td>7.21%</td><td></td><td>135.74</td></tr><tr><td>Adv-MLPB</td><td>0.1</td><td>4.02%</td><td>9.03%</td><td>9.68%</td><td>15.41% [2]</td><td>9.68%</td><td>✓</td><td>3.69</td></tr><tr><td>SDPd-MLPA</td><td>0.1</td><td>4.18%</td><td>11.51%</td><td>14.36%</td><td>34.77% [3]</td><td>30.81%</td><td></td><td>312.43</td></tr><tr><td>LPd-CNN</td><td>0.2</td><td>4.23%</td><td>9.54%</td><td>10.68%</td><td>17.50% [1]</td><td>10.68%</td><td>✓</td><td>7.32</td></tr><tr><td>LPd-CNN</td><td>0.3</td><td>11.40%</td><td>22.70%</td><td>25.79%</td><td>35.03% [1]</td><td>25.79%</td><td>✓</td><td>5.13</td></tr><tr><td>LPd-CNN</td><td>0.4</td><td>26.13%</td><td>39.22%</td><td>48.98%</td><td>62.49% [1]</td><td>48.98%</td><td>✓</td><td>5.07</td></tr><tr><td>CIFAR-10</td><td>LPd-RES</td><td>8/255</td><td>72.93%</td><td>76.52%</td><td>77.29%</td><td>78.52% [1]</td><td>77.60%</td><td></td><td>15.23</td></tr></table>

While performance depends on the training method and architecture, we improve on both the lower and upper bounds for every network tested. For lower bounds, we successfully find an adversarial

example for every test sample that PGD finds an adversarial example for. In addition, we observe that PGD 'misses' some valid adversarial examples: it fails to find these adversarial examples even though they are within the norm bounds. As the last three rows of Table 2 show, PGD misses for a larger fraction of test samples when  $\epsilon$  is larger. We also found that PGD is far more likely to miss for some test sample if the minimum adversarial distortion for that sample is close to  $\epsilon$ ; this observation is discussed in more depth in Appendix F. For upper bounds, we improve on the bound on adversarial error even when the upper bound on the worst-case loss — which is used to generate the certificate of robustness — is explicitly optimized for during training (as is the case for  $\mathrm{LP_d}$  and  $\mathrm{SDP_d}$  training). Our method also scales well to the more complex CIFAR-10 dataset and the larger  $\mathrm{LP_d}$ -RES network (which has 107,496 units), with the solver reaching the time limit for only  $0.31\%$  of samples.

Most importantly, we are able to determine the exact adversarial accuracy for  $\mathrm{Adv - MLP_B}$  and  $\mathrm{LP_d - CNN}$  for all  $\epsilon$  tested, finding either a certificate of robustness or an adversarial example for every test sample. For  $\mathrm{Adv - MLP_B}$  and  $\mathrm{LP_d - CNN}$ , running our verifier over the full test set takes approximately 10 hours — the same order of magnitude as the time to train each network on a single GPU. Better still, verification of individual samples is fully parallelizable.

# 5.2.1 OBSERVATIONS ON DETERMINANTS OF VERIFICATION TIME

Ceteris paribus, we might expect verification time to be correlated to the total number of ReLUs, since the solver may need to explore both possibilities for the phase of each ReLU. However, there is clearly more at play: even though  $\mathrm{LP_d}$ -CNN and Adv-CNN have identical architectures, verification time for Adv-CNN is two orders of magnitude higher.

Table 3: Determinants of verification time: mean verification time is 1) inversely correlated to the number of labels that can be eliminated from consideration and 2) correlated to the number of ReLUs that are not provably stable. Results are for  $\epsilon = 0.1$  on MNIST.  

<table><tr><td rowspan="3">Network</td><td rowspan="3">Mean Time / s</td><td rowspan="3">Number of Labels Eliminated</td><td colspan="4">Number of ReLUs</td></tr><tr><td rowspan="2">Possibly Unstable</td><td colspan="2">Provably Stable</td><td rowspan="2">Total</td></tr><tr><td>Active</td><td>Inactive</td></tr><tr><td>LPd-CNN</td><td>3.52</td><td>6.57</td><td>121.18</td><td>1552.52</td><td>3130.30</td><td>4804</td></tr><tr><td>Adv-CNN</td><td>135.74</td><td>3.14</td><td>545.90</td><td>3383.30</td><td>874.80</td><td>4804</td></tr><tr><td>Adv-MLPb</td><td>3.69</td><td>4.77</td><td>55.21</td><td>87.31</td><td>257.48</td><td>400</td></tr><tr><td>SDPd-MLPa</td><td>312.43</td><td>0.00</td><td>297.66</td><td>73.85</td><td>128.50</td><td>500</td></tr></table>

The key lies in the restricted input domain  $\mathcal{G}(x)$  for each test sample  $x$ . When input is restricted to  $\mathcal{G}(x)$ , we can prove that many ReLUs are stable (with respect to  $\mathcal{G}$ ). Furthermore, we can eliminate some labels from consideration by proving that the upper bound on the output neuron corresponding to that label is lower than the lower bound for some other output neuron. As the results in Table 3 show, a significant number of ReLUs can be proven to be stable, and a significant number of labels can be eliminated from consideration. Rather than being correlated to the total number of ReLUs, solve times are instead more strongly correlated to the number of ReLUs that are not provably stable, as well as the number of labels that cannot be eliminated from consideration.

# 6 DISCUSSION

This paper presents an efficient complete verifier for piecewise-linear neural networks. While we have focused on evaluating networks on the class of perturbations they are designed to be robust to, defining a class of perturbations that generates images perceptually similar to the original remains an important direction of research. Our verifier is able to handle new classes of perturbations (such as convolutions) as long as the set of perturbed images is a union of polytopes in the input space.

We close with ideas on improving verifiability of neural networks. As previously discussed, increasing the number of locally stable RLUs speeds up verification. We also observed (see Appendix G) that sparsifying weights promotes verifiability. Adopting a principled sparsification approach (for example,  $l_{1}$  regularization during training, or pruning and retraining (Han et al., 2016)) could potentially further increase verifiability without compromising on the true adversarial accuracy.

# REFERENCES

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In Proceedings of the 35th International Conference on Machine Learning, 2018.  
Osbert Bastani, Yani Ioannou, Leonidas Lampropoulos, Dimitrios Vytiniotis, Aditya Nori, and Antonio Criminisi. Measuring neural net robustness with constraints. In Advances in Neural Information Processing Systems, pp. 2613-2621, 2016.  
Jeff Bezanson, Alan Edelman, Stefan Karpinski, and Viral B Shah. Julia: A fresh approach to numerical computing. SIAM Review, 59(1):65-98, 2017.  
Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.  
Rudy Bunel, Ilker Turkaslan, Philip HS Torr, Pushmeet Kohli, and M Pawan Kumar. Piecewise linear neural network verification: A comparative study. arXiv preprint arXiv:1711.00455, 2017.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, pp. 3-14, 2017a.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In Security and Privacy (SP), 2017 IEEE Symposium on, pp. 39-57. IEEE, 2017b.  
Nicholas Carlini, Guy Katz, Clark Barrett, and David L Dill. Ground-truth adversarial examples. arXiv preprint arXiv:1709.10207, 2017.  
Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, and Cho-Jui Hsieh. EAD: elastic-net attacks to deep neural networks via adversarial examples. In AAAI Conference on Artificial Intelligence, 2018.  
Chih-Hong Cheng, Georg Nuhrenberg, and Harald Ruess. Maximum resilience of artificial neural networks. In International Symposium on Automated Technology for Verification and Analysis, pp. 251-268. Springer, 2017.  
Iain Dunning, Joey Huchette, and Miles Lubin. Jump: A modeling language for mathematical optimization. SIAM Review, 59(2):295-320, 2017. doi: 10.1137/15M1020575.  
Souradeep Dutta, Susmit Jha, Sriram Sankaranarayanan, and Ashish Tiwari. Output range analysis for deep feedforward neural networks. In NASA Formal Methods Symposium, pp. 121-138. Springer, 2018.  
Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy Mann, and Pushmeet Kohli. A dual approach to scalable verification of deep networks. In Conference on Uncertainty in Artificial Intelligence, 2018.  
Ruediger Ehlers. Formal verification of piece-wise linear feed-forward neural networks. In International Symposium on Automated Technology for Verification and Analysis, pp. 269-286. Springer, 2017.  
Matteo Fischetti and Jason Jo. Deep neural networks and mixed integer linear optimization. Constraints, 23(3):296-309, July 2018. ISSN 1383-7133. doi: 10.1007/s10601-018-9285-6. URL https://doi.org/10.1007/s10601-018-9285-6.  
Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, and Martin Vechev. Ai 2: Safety and robustness certification of neural networks with abstract interpretation. In Security and Privacy (SP), 2018 IEEE Symposium on, 2018.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.  
Gurobi. Gurobi guidelines for numerical issues, 2017. URL http://files.gurobi.com/ Numerics.pdf.

Inc. Gurobi Optimization. Gurobi optimizer reference manual, 2017. URL http://www.gurobi.com.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Matthias Hein and Maksym Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. In Advances in Neural Information Processing Systems, pp. 2263-2273, 2017.  
Xiaowei Huang, Marta Kwiatkowska, Sen Wang, and Min Wu. Safety verification of deep neural networks. In International Conference on Computer Aided Verification, pp. 3-29. Springer, 2017.  
Joey Huchette and Juan Pablo Vielma. Nonconvex piecewise linear functions: Advanced formulations and simple modeling tools. arXiv preprint arXiv:1708.00050, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, 2015.  
Guy Katz, Clark Barrett, David L Dill, Kyle Julian, and Mykel J Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In International Conference on Computer Aided Verification, pp. 97-117. Springer, 2017.  
J Zico Kolter and Eric Wong. Provable defenses against adversarial examples via the convex outer adversarial polytope. In Proceedings of the 35th International Conference on Machine Learning, 2017.  
Alessio Lomuscio and Lalit Maganti. An approach to reachability analysis for feed-forward relu neural networks. arXiv preprint arXiv:1706.07351, 2017.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.  
Ramon E Moore, R Baker Kearfott, and Michael J Cloud. Introduction to interval analysis. SIAM, 2009.  
Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, and Ananthram Swami. Distillation as a defense to adversarial perturbations against deep neural networks. In *Security and Privacy (SP)* 2016 IEEE Symposium on, pp. 582-597. IEEE, 2016.  
Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. In International Conference on Learning Representations, 2018.  
Karsten Scheibler, Leonore Winterer, Ralf Wimmer, and Bernd Becker. Towards verification of artificial neural networks. In  $MBMV$ , pp. 30-40, 2015.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014.  
Juan Pablo Vielma. Mixed integer linear programming formulation techniques. SIAM Review, 57(1): 3-57, 2015.

Tsui-Wei Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane Boning, Inderjit S Dhillon, and Luca Daniel. Towards fast computation of certified robustness for relu networks. In Proceedings of the 35th International Conference on Machine Learning, 2018.  
Eric Wong, Frank Schmidt, Jan Hendrik Metzen, and J Zico Kolter. Scaling provable adversarial defenses. arXiv preprint arXiv:1805.12514, 2018.  
Weiming Xiang, Hoang-Dung Tran, and Taylor T. Johnson. Output reachable set estimation and verification for multi-layer neural networks. IEEE Transactions on Neural Networks and Learning Systems (TNNLS), March 2018.
