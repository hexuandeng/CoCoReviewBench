# AN EFFICIENT AND MARGIN-APPROACHING ZERO-CONFIDENCE ADVERSARIAL ATTACK

Anonymous authors

Paper under double-blind review

# ABSTRACT

There are two major paradigms of white-box adversarial attacks that attempt to impose input perturbations. The first paradigm, called the fix-perturbation attack, crafts adversarial samples within a given perturbation level. The second paradigm, called the zero-confidence attack, finds the smallest perturbation needed to cause misclassification, also known as the margin of an input feature. While the former paradigm is well-resolved, the latter is not. Existing zero-confidence attacks either introduce significant approximation errors, or are too time-consuming. We therefore propose MARGINATTACK, a zero-confidence attack framework that is able to compute the margin with improved accuracy and efficiency. Our experiments show that MARGINATTACK is able to compute a smaller margin than the state-of-the-art zero-confidence attacks, and matches the state-of-the-art fix-perturbation attacks. In addition, it runs significantly faster than the Carlini-Wagner attack, currently the most accurate zero-confidence attack algorithm.

# 1 INTRODUCTION

Adversarial attack refers to the task of finding small and imperceptible input transformations that cause a neural network classifier to misclassify. White-box attacks are a subset of attacks that have access to gradient information of the target network. In this paper, we will focus on the white-box attacks. An important class of input transformations is adding small perturbations to the input. There are two major paradigms of adversarial attacks that attempt to impose input perturbations. The first paradigm, called the fix-perturbation attack, tries to find perturbations that are most likely to cause misclassification, with the constraint that the norm of the perturbations cannot exceed a given level. Since the perturbation level is fixed, fix-perturbation attacks may fail to find any adversarial samples for inputs that are far away from the decision boundary. The second paradigm, called the zero-confidence attack, tries to find the smallest perturbations that are guaranteed to cause misclassification, regardless of how large the perturbations are. Since they aim to minimize the perturbation norm, zero-confidence attacks usually find adversarial samples that ride right on the decision boundaries, and hence the name "zero-confidence". The resulting perturbation norm is also known as the margin of an input feature to the decision boundary. Both of these paradigms are essentially constrained optimization problems. The former has a simple convex constraint (perturbation norm), but a non-convex target (classification loss or logit differences). In contrast, the latter has a non-convex constraint (classification loss or logit differences), but a simple convex target (perturbation norm).

Despite their similarity as optimization problems, the two paradigms differ significantly in terms of difficulty. The fix-perturbation attack problem is easier. The state-of-the-art algorithms, including projected gradient descent (PGD) (Madry et al., 2017) and distributional adversarial attack (Zheng et al., 2018), can achieve both high efficiency and high success rate, and often come with theoretical convergence guarantee. On the other hand, the zero-confidence attack problem is much more challenging. Existing methods are either not strong enough or too slow. For example, DeepFool (Moosavi Dezfooli et al., 2016) and fast gradient sign method (FGSM) (Goodfellow et al., 2014; Kurakin et al., 2016a;b) linearizes the constraint, and solves the simplified optimization problem with a simple convex target and a linear constraint. However, due to the linearization approximation errors, the solution can be far from optimal. As another extreme, L-BFGS (Szegedy et al., 2013) and Carlini-Wagner (CW) (Carlini & Wagner, 2017) convert the optimization problem into a Lagrangian, and the Lagrangian multiplier is determined through grid search or binary search. These attacks are generally much stronger and theoretically grounded, but can be very slow.

The necessity of developing a better zero-confidence attack is evident. The zero-confidence attack paradigm is a more realistic attack setting. More importantly, it aims to measure the margin of each individual token, which lends more insight into the data distribution and adversarial robustness. Motivated by this, we propose MARGINATTACK, a zero-confidence attack framework that is able to compute the margin with improved accuracy and efficiency. Specifically, MARGINATTACK iterates between two moves. The first move, called restoration move, linearizes the constraint and solves the simplified optimization problem, just like DeepFool and FGSM; the second move, called projection move, explores even smaller perturbations without changing the constraint values significantly. By construction, MARGINATTACK inherits the efficiency in DeepFool and FGSM, and improves over them in terms of accuracy with a convergence guarantee. Our experiments show that MARGINATTACK attack is able to compute a smaller margin than the state-of-the-art zero-confidence attacks, and matches the state-of-the-art fix-perturbation attacks. In addition, it runs significantly faster than CW, and in some cases comparable to DeepFool and FGSM.

# 2 RELATED WORKS

In addition to the aforementioned state-of-the-art attacks, there are a couple of other works that attempt to explore the margin. Jacobian-based saliency map attack (Papernot et al., 2016) is among the earliest works that apply gradient information to guide the crafting of adversarial examples. It chooses to perturb the input features whose gradient is consistent with the adversarial goal. One-pixel attack (Su et al., 2017) finds adversarial examples by perturbing only one pixel, which can be regarded as finding the  $\ell_0$  margin of the inputs. Ilyas et al. (2018) converts PGD into a zero-confidence attack by searching different perturbation levels, but this again can be time-consuming because it needs to solve multiple optimization subproblems. Weng et al. proposed a metric called CLEVER (Weng et al., 2018), which estimates an upper-bound of the margins. Unfortunately, recent work (Goodfellow, 2018) has shown that CLEVER can overestimate the margins due to gradient masking (Papernot et al., 2017). The above are a just a small subset of white-box attack algorithms that are relevant to our work. For an overview of the field, we refer readers to Akhtar & Mian (2018).

The MARGINATTACK framework is inspired by the Rosen's algorithm (Rosen, 1961) for constraint optimization problems. However, there are several important distinctions. First, the Rosen's algorithm rests on some unrealistic assumptions for neural networks, e.g. continuously differentiable constraints, while MARGINATTACK has a convergence guarantee with a more realistic set of assumptions. Second, the Rosen's algorithm requires a step size search for each iteration, which can be time-consuming, whereas MARGINATTACK will work with a simple diminishing step size scheme. Most importantly, as will be shown later, MARGINATTACK refers to a large class of attack algorithms depending on how the two parameters,  $a^{(k)}$  and  $b^{(k)}$ , are set, and the Rosen's algorithm only fits into one of the settings, which only works well under the  $\ell_2$  norm. For other norms, there exist other parameter settings that are much more effective. As another highlight, the convergence guarantee of MARGINATTACK holds for all the settings that satisfy some moderate assumptions.

# 3 THE MARGINATTACK ALGORITHM

In this section, we will formally introduce the algorithm and discuss its convergence properties. In the paper, we will denote scalars with non-bolded letters, e.g.  $a$  or  $A$ ; column vectors with lower-cased, bolded letters, e.g.  $\mathbf{a}$ ; matrix with upper-cased, bolded letters, e.g.  $\mathbf{A}$ ; sets with upper-cased double-stoke letters, e.g.  $\mathbb{A}$ ; gradient of a function  $f(\pmb{x})$  evaluated at  $\pmb{x} = \pmb{x}_0$  as  $\nabla f(\pmb{x}_0)$ .

# 3.1 PROBLEM FORMULATION

Given a classifier whose output logits are denoted as  $l_{0}(\pmb{x}), l_{1}(\pmb{x}), \dots, l_{C-1}(\pmb{x})$ , where  $C$  is the total number of classes, for any data token  $(\pmb{x}_{0}, t)$ , where  $\pmb{x}_{0}$  is an  $n$ -dimensional input feature vector, and  $t \in \{0, \dots, C-1\}$  is its label, MARGINATTACK computes

$$
\boldsymbol {x} ^ {*} = \underset {\boldsymbol {x}} {\arg \min } d (\boldsymbol {x} - \boldsymbol {x} _ {0}), \text {s . t .} c (\boldsymbol {x}) \leq 0, \tag {1}
$$

where  $d(\cdot)$  is a norm. In this paper we only consider  $\ell_2$  and  $\ell_{\infty}$  norms, but the proposed method is generalizable to other norms. For non-targeted adversarial attacks, the constraint is defined as

$$
c (\boldsymbol {x}) = l _ {t} (\boldsymbol {x}) - \max  _ {i \neq t} l _ {i} (\boldsymbol {x}) - \varepsilon , \tag {2}
$$

where  $\varepsilon$  is the offset parameter. As a common practice,  $\varepsilon$  is often set to a small negative number to ensure that the adversarial sample lies on the incorrect side of the decision boundary. In this paper, we will only consider non-targeted attack, but all the discussions are applicable to targeted attacks (i.e.  $c(\pmb{x}) = \max_{i \neq a} l_i(\pmb{x}) - l_a(\pmb{x}) - \varepsilon$  for a target class  $a$ ).

# 3.2 THE MARGINATTACK PROCEDURE

MARGINATTACK alternately performs the restoration move and the projection move. Specifically, denote the solution after the  $k$ -th iteration as  $\pmb{x}^{(k)}$ . Then the two steps are:

Restoration Move: The restoration move tries to hop to the constraint boundary, i.e.  $c(\pmb{x}) = 0$  with the shortest hop. Formally, it solves:

$$
\boldsymbol {z} ^ {(k)} = \underset {\boldsymbol {x}} {\arg \min } d (\boldsymbol {x} - \boldsymbol {x} ^ {(k)}), \quad \text {s . t .} \nabla^ {T} c \left(\boldsymbol {x} ^ {(k)}\right) \left(\boldsymbol {x} - \boldsymbol {x} ^ {(k)}\right) = - \alpha^ {(k)} c \left(\boldsymbol {x} ^ {(k)}\right). \tag {3}
$$

where  $\alpha^{(k)}$  is the step size within [0, 1]. Notice that the left hand side of the constraint in Eq. (3) is the first-order Taylor approximation of  $c(\pmb{z}^{(k)}) - c(\pmb{x}^{(k)})$ , so this constraint tries to move point closer to  $c(\pmb{x}) = 0$  by  $\alpha^{(k)}$ . It can be shown, from the dual-norm theory, that the solution to (3) is

$$
\boldsymbol {z} ^ {(k)} = \boldsymbol {x} ^ {(k)} - \frac {\alpha^ {(k)} c (\boldsymbol {x} ^ {(k)}) \boldsymbol {s} (\boldsymbol {x} ^ {(k)})}{\nabla^ {T} c (\boldsymbol {x} ^ {(k)}) \boldsymbol {s} (\boldsymbol {x} ^ {(k)})} \tag {4}
$$

$\pmb{s}(\pmb{x})$  is defined such that  $\nabla^T c(\pmb{x})\pmb{s}(\pmb{x}) = d^*(\nabla^T c(\pmb{x}))$ , where  $d^*(\cdot)$  is the dual norm of  $d(\cdot)$ . Specifically, noticing that the dual norm of the  $\ell_p$  norm is the  $\ell_{(1 - p^{-1})^{-1}}$  norm, we have

$$
s (\boldsymbol {x}) = \left\{ \begin{array}{l l} \nabla c (\boldsymbol {x}) / \| \nabla c (\boldsymbol {x}) \| _ {2} & \text {i f} d (\cdot) \text {i s t h e} \ell_ {2} \text {n o r m} \\ \operatorname {s i g n} (\nabla c (\boldsymbol {x})) & \text {i f} d (\cdot) \text {i s t h e} \ell_ {\infty} \text {n o r m} \end{array} \right.. \tag {5}
$$

As mentioned, Eq. (4) is similar to DeepFool under  $\ell_2$  norm, and to FGSM under  $\ell_{\infty}$  norm. Therefore, we can expect that the restoration move should effectively hop towards the decision boundary, but the hop direction may not be optimal. That is why we need the next move.

Projection Move: The projection move tries to move closer to  $\pmb{x}_0$  while ensuring that  $c(\pmb{x})$  will not change drastically. Formally,

$$
\boldsymbol {x} ^ {(k + 1)} = \boldsymbol {z} ^ {(k)} - \beta^ {(k)} a ^ {(k)} \nabla d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right) - \beta^ {(k)} b ^ {(k)} \boldsymbol {s} \left(\boldsymbol {z} ^ {(k)}\right) \tag {6}
$$

where  $\beta^{(k)}$  is the step size within [0,1];  $a^{(k)}$  and  $b^{(k)}$  are two scalars, which will be specified later. As an intuitive explanation on Eq. (3), notice that the second term, which we will call the distance reduction term, reduces the distance to  $x_0$ , whereas the third term, which we will call the constraint reduction term, reduces the constraint (because  $s(z^{(k)})$  and  $\nabla c(z^{(k)})$  has a positive inner product). Therefore, the projection move essentially strikes a balance between reduction in distance and reduction in constraint.

$a^{(k)}$  and  $b^{(k)}$  can have two designs. The first design is to ensure the constraint values are roughly the same after the move, i.e.  $c(\pmb{z}^{(k)}) - c(\pmb{x}^{(k + 1)}) \approx 0$ . By Taylor approximation, we have

$$
\nabla^ {T} c \left(\boldsymbol {z} ^ {(k)}\right) \left(\boldsymbol {x} ^ {(k + 1)} - \boldsymbol {z} ^ {(k)}\right) = 0, \tag {7}
$$

whose solution is

$$
b ^ {(k)} = \frac {a ^ {(k)} \nabla^ {T} c \left(\boldsymbol {z} ^ {(k)}\right) \nabla d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right)}{\nabla^ {T} c \left(\boldsymbol {z} ^ {(k)}\right) \boldsymbol {s} \left(\boldsymbol {z} ^ {(k)}\right)}. \tag {8}
$$

Another design is to ensure the perturbation norm reduces roughly by  $\beta^{(k)}$ , i.e.  $d(\pmb{x}^{(k + 1)} - \pmb{x}_0) \approx (1 - \beta^{(k)}) d(\pmb{z}^{(k)} - \pmb{x}_0)$ . By Taylor approximation, we have

$$
\nabla^ {T} d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right) \left(\boldsymbol {x} ^ {(k + 1)} - \boldsymbol {z} ^ {(k)}\right) = \beta^ {(k)} d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right), \tag {9}
$$

whose solution is

$$
a ^ {(k)} = 1 - \frac {b ^ {(k)} \nabla^ {T} d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right) \boldsymbol {s} \left(\boldsymbol {z} ^ {(k)}\right)}{\nabla^ {T} d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right) \nabla d \left(\boldsymbol {z} ^ {(k)} - \boldsymbol {x} _ {0}\right)} \tag {10}
$$

It should be noted that Eqs. (8) and (10) are just two specific choices for  $a^{(k)}$  and  $b^{(k)}$ . It turns out that MARGINATTACK will work with a convergence guarantee for a wide range of bounded  $a^{(k)}\mathrm{s}$  and  $b^{(k)}\mathrm{s}$  that satisfy some conditions, as will be shown in section 3.4. Therefore, MARGINATTACK provides a general and flexible framework for zero-confidence adversarial attack designs. In practice, we find that Eq. (8) works better for  $\ell_2$  norm, and Eq. (8) works better for  $\ell_{\infty}$  norm.

# 3.3 HOW MARGINATTACK WORKS

Figure 1 illustrates a typical convergence path of MARGINATTACK using  $\ell_2$  norm and Eq. (8) as an example. The red dots on the right denote the original inputs  $x_0$  and its closest point on the decision boundary,  $x^*$ . Suppose after iteration  $k$ , MARGINATTACK reaches  $x^{(k)}$ , denoted by the green dot on the left. The restoration move travels directly towards the decision boundary by finding the normal direction to the current constraint contour. Then, the projection move travels along the tangent plane of the current constraint contour to reduce the distance to  $x_0$  while preventing the constraint value from deviating much. As intuitively expected, the iteration should eventually approach  $x^*$ . Figure 2 plots an empirical convergence curve of the perturbation norm and constraint value of MARGINATTACK- $\ell_2$  on a randomly chosen CIFAR image. Each move from a triangle to a circle dot is a restoration move, and from circle to triangle a projection move. The red line is the smoothed version. As can be seen, a restoration move reduces the constraint value while slightly increasing the constraint norm, and a projection move reduces the perturbation norm while slightly affecting the constraint value. Both curves can eventually converge.

# 3.4 THE CONVERGENCE GUARANTEE

The constraint function  $c(\pmb{x})$  in Eq. (2) is nonconvex, thus the convergence analysis for MARGINAT-TACK is limited to the vicinity of a unique local optimum, as stated in the following theorem.

Theorem 1. Denote  $\mathbf{x}^*$  as one local optimum for Eq. (1). Assume  $\nabla c(\mathbf{x}^*)$  exists. Define projection matrices

$$
\boldsymbol {P} = \boldsymbol {I} - \boldsymbol {s} \left(\boldsymbol {x} ^ {*}\right) \left(\nabla^ {T} c \left(\boldsymbol {x} ^ {*}\right) \boldsymbol {s} \left(\boldsymbol {x} ^ {*}\right)\right) ^ {- 1} \nabla^ {T} c \left(\boldsymbol {x} ^ {*}\right) \tag {11}
$$

Consider the neighborhood  $\mathbb{B} = \{\pmb{x} : \| \pmb{P}[\pmb{x} - \pmb{x}^*] \|_2^2 \leq X, |c(\pmb{x})| \leq C\}$  that satisfies the following assumptions:

1. (Differentiability)  $\forall \pmb{x} \in \mathbb{B}, \nabla c(\pmb{x})$  exists, but can be discontinuous, i.e., all the discontinuity points of the gradient in  $\mathbb{B}$  are jump discontinuities;  
2. (Lipschitz Continuity at  $\pmb{x}^*$ )  $\forall \pmb{x} \in \mathbb{B}$ ,  $\| s(\pmb{x}) - s(\pmb{x}^*) \|_2 \leq L_s \| s(\pmb{x}^*) \|_2 \| \pmb{x} - \pmb{x}^* \|_2$ ;  
3. (Bounded Gradient Norm)  $\forall \pmb {x}\in \mathbb{B},0 <   m\leq \| \nabla c(\pmb {x})\|_{2}\leq M;$  
4. (Bounded Gradient Difference)  $\exists \delta > 0, \forall \boldsymbol{x}, \boldsymbol{y} \in \mathbb{B}$  s.t.  $\boldsymbol{y} - \boldsymbol{x} = l\boldsymbol{s}(\boldsymbol{x})$  for some  $l$ ,

$$
\nabla^ {T} c (\boldsymbol {y}) \boldsymbol {s} (\boldsymbol {x}) \geq \delta \nabla^ {T} c (\boldsymbol {x}) \boldsymbol {s} (\boldsymbol {x});
$$

5. (Constraint Convexity)  $\exists \gamma \in (0,1),\forall \pmb {x}\in \mathbb{B},$

$$
\left(a ^ {(k)} \nabla d (\boldsymbol {x} - \boldsymbol {x} _ {0}) + b ^ {(k)} \boldsymbol {s} (\boldsymbol {x})\right) ^ {T} \boldsymbol {P} ^ {T} \boldsymbol {P} (\boldsymbol {x} - \boldsymbol {x} _ {0}) \geq \gamma (\boldsymbol {x} - \boldsymbol {x} _ {0}) ^ {T} \boldsymbol {P} ^ {T} \boldsymbol {P} (\boldsymbol {x} - \boldsymbol {x} _ {0});
$$

6. (Unique Optimality)  $\pmb{x}^*$  is the only global optimum within  $\mathbb{B}$ ;  
7. (Constant Bounded Restoration Step Size)  $\alpha^{(k)} = \alpha < M_{\alpha};^2$  
8. (Shrinking Projected Step Size)  $\beta^{(k)} < \beta / (k + k_0)^\nu$ , where  $0 < \nu < 1$  and  $\beta \leq M_\beta, k_0 > m_k$ ;  $|a^{(k)}| < M_a, |b^{(k)}| < M_b$ ;  
9. (Presence in Neighborhood)  $\exists K, \boldsymbol{x}^{(K)} \in \operatorname{int}[\mathbb{B}]$ , i.e. the interior of  $\mathbb{B}$ .

Then we have the convergence guarantee  $\lim_{k\to \infty}\| \pmb{x}^{(k)} - \pmb{x}^*\| _2 = 0$

The proof will be presented in the appendix. Here are a few remarks. First, assumption 1 allows jump discontinuities in  $\nabla c(\pmb{x})$  almost everywhere, which is a very practical assumption for deep neural networks. Most neural network operations, such as ReLU and max-pooling, as well as the max operation in Eq. (2), introduce nothing beyond jump discontinuities in gradient.

Second, assumption 3 does require the constraint gradient to be lower bounded, which may lead to concerns that MARGINATTACK may fail in the presence of gradient masking (Papernot et al., 2017). However, notice that the gradient boundedness assumption is only imposed in  $\mathbb{B}$ , which is in the vicinity of the decision boundary, whereas gradient masking is most likely to appear away from the decision boundary and where the input features are populated. Besides, as will be discussed later, a random initialization as in PGD will be adopted to bypass regions with gradient masking. Experiments on adversarially trained models also verify the robustness of MARGINATTACK.

Finally, assumption 5 essentially stipulates that  $c(\pmb{x})$  is convex or "not too concave" in  $\mathbb{B}$  (and thus so is the constraint set  $c(\pmb{x}) \leq 0$ ), so that the first order optimality condition can readily imply local minimum instead of a local maximum. In fact, it can be shown that assumption 5 can be implied if  $c(\pmb{x})$  is convex in  $\mathbb{B}$ .<sup>4</sup>

# 3.5 ADDITIONAL IMPLEMENTATION DETAILS

There are a few additional implementation details as outlined below.

Box Constraint: In many applications, each dimension of the input features should be bounded, i.e.  $\pmb{x} \in [x_{\min}, x_{\max}]^n$ . To impose the box constraint, the restoration move problem as in Eq. (3) is modified as

$$
\boldsymbol {z} ^ {(k)} = \underset {\boldsymbol {x} \in \left[ x _ {\min }, x _ {\max } \right] ^ {n}} {\arg \min } d \left(\boldsymbol {x} - \boldsymbol {x} ^ {(k)}\right), \quad \text {s . t .} \nabla^ {T} c \left(\boldsymbol {x} ^ {(k)}\right) \left(\boldsymbol {x} - \boldsymbol {x} ^ {(k)}\right) = - \alpha^ {(k)} c \left(\boldsymbol {x} ^ {(k)}\right), \tag {12}
$$

whose solution is

$$
\boldsymbol {z} ^ {(k)} = \operatorname {P r o j} _ {\left[ x _ {\min }, x _ {\max } \right] ^ {n}} \left\{\tilde {\boldsymbol {z}} ^ {(k)} \right\}, \text {w h e r e} \tilde {\boldsymbol {z}} ^ {(k)} = \boldsymbol {x} ^ {(k)} - \frac {\alpha^ {(k)} c \left(\boldsymbol {x} ^ {(k)}\right) + \sum_ {i \in \mathbb {I} ^ {C}} \nabla_ {i} c \left(\boldsymbol {x} ^ {(k)}\right) \left(\boldsymbol {z} _ {i} ^ {(k)} - \boldsymbol {x} _ {i} ^ {(k)}\right)}{\sum_ {i \in \mathbb {I}} \nabla_ {i} c \left(\boldsymbol {x} ^ {(k)}\right) \boldsymbol {s} _ {i} \left(\boldsymbol {x} ^ {(k)}\right)} \boldsymbol {s} \left(\boldsymbol {x} ^ {(k)}\right). \tag {13}
$$

$\operatorname{Proj}(\cdot)$  is an operator that projects the vector in its argument onto the subset in its subscript.  $\mathbb{I}$  is a set of indices with which the elements in  $\tilde{\boldsymbol{z}}^{(k)}$  satisfy the box constraint, and  $\mathbb{I}^C$  is its complement.  $\mathbb{I}$  is determined by running Eq. (13) iteratively and updating  $\mathbb{I}$  after each iterations.

Unlike other attack algorithms that simply project the solution onto the constraint box, MARGINATTACK incorporates the box constraint in a principled way, such that any local optimal solution  $\pmb{x}^*$  will be an invariant point of the restoration move. Thus the convergence is faster.

Target Scan: According to Eq. (2), each restoration move essentially approaches the adversarial class with the highest logit, but the class with the highest logit may not be the closest. To mitigate the problem, we follow a similar approach adopted in DeepFool, which we call target scan. Target scan performs a target-specific restoration move towards each class, and chooses the move with the shortest distance. Formally, target scan introduces a set of target-specific constraints  $\{c_i(\pmb{x}) = l_t(\pmb{x}) - l_i(\pmb{x}) - \varepsilon\}$ . A restoration move with target scan solves

$$
\boldsymbol {z} ^ {(k)} = \underset {i \in \mathbb {A}} {\arg \min } d \left(\boldsymbol {z} ^ {(k, i)} - \boldsymbol {x} _ {0}\right) \tag {14}
$$

where  $\pmb{z}^{(k,i)}$  is the solution to Eqs. (3) or (12) with  $c(\pmb{x}^{(k)})$  replaced with  $c_{i}(\pmb{x}^{(k)})$ , and thus is equal to Eqs. (4) or (13) with  $c(\pmb{x}^{(k)})$  replaced with  $c_{i}(\pmb{x}^{(k)})$ .  $\mathbb{A}$  is a set of candidate adversarial calsses, which can be all the incorrect classes if the number of classes is small, or which can be a subset of the adversarial classes with the highest logits otherwise. Experiments show that target scan is necessary only in the first few restoration moves, when the closest and highest adversarial classes are likely to be distinct. Therefore, the computation cost will not increase too much.

Initialization: The initialization of  $\pmb{x}^{(0)}$  can be either deterministic or random as follows

$$
\boldsymbol {x} ^ {(0)} = \boldsymbol {x} _ {0} (\text {D e t e r m i n i s t i c}), \quad \boldsymbol {x} ^ {(0)} = \boldsymbol {x} _ {0} + \boldsymbol {u}, \quad \boldsymbol {u} \sim \mathcal {U} \left\{\left[ - u, u \right] ^ {n} \right\} (\text {R a n d o m}) \tag {15}
$$

where  $\mathcal{U}\{[-u,u]^n\}$  denotes the uniform random distribution in  $[-u,u]^n$ . Similar to PGD, we can perform multiple trials with random initialization to find a better local optimum.

Final Tuning MARGINATTACK can only cause misclassification when  $c(\pmb{x}) \leq \varepsilon$ . To make sure the attack is successful, the final iterations of MARGINATTACK consists of restoration moves only,

![](images/a23de397cd313f6c0b795a7d7def2e8a6c048b93bfe3f5f50620c13e5ca9ada6.jpg)  
Figure 1: A convergence path of MARGINATTACK.

![](images/32cc89b800131aaaa86410795a7d774bd9311adae6b0f84e48e1f812eecdd12a.jpg)  
Figure 2: An empirical convergence curve of perturbation norm (left) and constraint value (right).

![](images/998a6e03b2f3aa74c7e06cf6244cf79636d705853c94e221fd653b4c4bdd97d2.jpg)

Algorithm 1: MARGINATTACK Procedure  
Input: A set of logit functions  $l_{0:C-1}(\pmb{x})$ ; an input feature  $\pmb{x}_0$  and its label  $t$ ; Output: A solution  $\tilde{\pmb{x}}^*$  to Eq. (1) Initialize  $\pmb{x}^{(0)}$  according to Eq. (15); for  $k <$  number of iterations do if  $k <$  number of target scan iterations then Do target scan restoration move as in Eq. (14); else Do regular restoration move as in Eqs. (3) or (12); end if  $k <$  final tuning iteration then Do projection move as in Eqs. (6); else Skip projection move:  $\pmb{x}^{(k+1)} = \pmb{z}^{(k)}$ ; end  
 $\tilde{\pmb{x}}^* = \pmb{x}^{(k)}$ .

and no projection moves, until a misclassification is caused. This can also ensure the final solution satisfies the box constraint (because only the restoration move incorporates the box constraint).

Summary: Alg. 1 summarizes the MARGINATTACK procedure. As for the complexity, each restoration move or projection move requires only one backward propagation, and thus the computational complexity of each move is comparable to one iteration of most attack algorithms.

# 4 EXPERIMENTS

This section compares MARGINATTACK with several state-of-the-art adversarial attack algorithms in terms of the perturbation norm and computation time on image classification benchmarks.

# 4.1 ATTACKING REGULAR MODELS

# 4.1.1 CONFIGURATIONS

Three regularly trained models are evaluated on.

- MNIST (LeCun et al., 1998): The classifier is a stack of two  $5 \times 5$  convolutional layers with 32 and 64 filters respectively, followed by two fully-connected layers with 1,024 hidden units.  
- CIFAR10 (Krizhevsky & Hinton, 2009): The classifier is a pre-trained ResNet32 (He et al., 2016) provided by TensorFlow.5.  
- ImageNet (Russakovsky et al., 2015): The classifier is a pre-trained ResNet50 (He et al., 2016) provided by TensorFlow Keras<sup>6</sup>. Evaluation is on a validation subset containing 10,000 images.

The range of each pixel is [0, 1] for MNIST, and [0, 255] for CIFAR10 and ImageNet. The settings of MARGINATTACK and baselines are listed below. Unless stated otherwise, the baseline algorithms are implemented by cleverhans (Nicolas Papernot, 2017). The hyperparameters are set to defaults if not specifically stated.

- CW (Carlini & Wagner, 2017): The target and evaluation norm is  $\ell_2$ . The learning rate is set to 0.05 for MNIST, 0.001 for CIFAR10 and 0.01 for ImageNet, which are tuned to its best performance. The number of binary steps for multiplier search is 10.

![](images/3fa4e5e4e0480bc9dca64356a5ba79d8526dbc07e38947986c54ce67a9c2d875.jpg)

![](images/6c5d9fe8152c94c85cd5821dbafa43d5d14c70bbd5b6c5c94b762c601f59cdcf.jpg)

![](images/b8ffcd1b04dceac6003aa2cebf23af4ba9159fe45bac9ade9e107dba65d4c6da.jpg)

![](images/96bf085cdbf7535c91f0b2621bd9e156ddfaff6c7f2f3ab93078346a23ae4d3f.jpg)  
Figure 3: Adversarial attacks on (left) MNIST, (middle) Cifar, and (right) ImageNet dataset.

![](images/23342910679d4258a1891b21009676e1a54b4a4ae8899a6defe8dc7065771103.jpg)

![](images/63f66f40cbce2250ec02e9ffb927cfe5ecf0830b87dc7929857e403ff6d8a21e.jpg)

- DeepFool (Moosavi Dezfooli et al., 2016): The evaluation norm is  $\ell_2$ .  
- FGSM (Goodfellow et al., 2014): FGSM is implemented by authors. The step size is searched to achieve zero-confidence attack. The evaluation distance metric is  $\ell_{\infty}$ .  
- PGD (Madry et al., 2017): The target and evaluation norm are  $\ell_{\infty}$ . The learning rate is set to 0.01 for MNIST, and 0.05 for CIFAR10 and 0.1 for ImageNet.  
- MARGINATTACK: Two versions of MARGINATTACK are implemented, whose target and evaluation norms are  $\ell_2$ , and  $\ell_{\infty}$ , respectively. The hyperparameters are detailed in Table 4 in the appendix. The first 10 restoration moves are with target scan, and the last 20 moves are all restoration moves.

The number of iterations/moves is set to 2,000 for CW, 200 with 10 random starts for PGD and MARGINATTACK (except for ImageNet where there is only one random run), and 200 for the rest.

# 4.1.2 RESULTS AND ANALYSES

Except for PGD, all the other attacks are zero-confidence attacks. For these attacks, we plot the CDF of the margins of the validation data, which can also be interpreted as the percentage success rate of these attacks as a function of perturbation level. Figure 3 plots the success rate curves, where the upper panel shows the  $\ell_2$  attacks, and the lower one shows  $\ell_{\infty}$  attacks. As can be observed, the MARGINATTACK curves are above all other algorithms at all perturbation levels and in all datasets. CW is very close to MARGINATTACK on MNIST and CIFAR10, but MARGINATTACK maintains a  $3\%$  advantage on MNIST and  $1\%$  on CIFAR10. It seems that CW is unable to converge well within 2,000 iterations on ImageNet, although the learning rate has been tuned to maximize its performance. MARGINATTACK, on the other hand, converges more efficiently and consistently.

To obtain a success rate curve for PGD, we have to run the attack again and again for many different perturbation levels, which can be time-consuming for large datasets (this shows an advantage of zero-confidence attacks over fix-perturbation attacks). Instead, we choose four perturbation levels for each attack scenario to compare. The perturbation levels are chosen to roughly follow the 0.2, 0.4, 0.6 and 0.8 quantiles of the MARGINATTACK margins. Table 1 compares the success rates under the chosen quantiles among the  $\ell_{\infty}$  attacks. We can see that MARGINATTACK outperforms PGD under all the perturbation levels, and that both significantly dominate FGSM.

# 4.2 ATTACKING ADVERSARIALLY TRAINED MODEL

We also evaluate MARGINATTACK on the MNIST Adversarial Examples Challenge<sup>7</sup>, which is a challenge of attacking an MNIST model adversarially trained using PGD with 0.3 perturbation level.

Table 1: Success rate (%) of adversarial attacks under given perturbation norms.  

<table><tr><td rowspan="2">Algorithm</td><td>MNIST</td><td>CIFAR</td><td>IMAGENET</td></tr><tr><td>0.06 / 0.08 / 0.10 / 0.12</td><td>0.2 / 0.4 / 0.6 / 1</td><td>0.05 / 0.1 / 0.2 / 0.4</td></tr><tr><td>FGSM</td><td>7.55 / 13.9 / 24.9 / 35.4</td><td>18.5 / 31.0 / 41.1 / 54.7</td><td>39.8 / 47.2 / 60.1 / 75.3</td></tr><tr><td>PGD</td><td>17.1 / 42.2 / 73.7 / 91.8</td><td>18.9 / 38.9 / 59.1 / 84.1</td><td>40.4 / 49.8 / 68.8 / 90.6</td></tr><tr><td>Ours</td><td>18.1 / 43.0 / 74.1 / 92.1</td><td>21.1 / 42.2 / 62.6 / 87.3</td><td>41.5 / 51.3 / 69.0 / 90.8</td></tr></table>

Table 2: Success rate under 0.3 perturbation norm of the MNIST Adversarial Examples Challenge.  

<table><tr><td>Algorithm</td><td>Success Rate (%)</td></tr><tr><td>Zheng et al. (2018)</td><td>11.21</td></tr><tr><td>MARGINATTACK (l∞)</td><td>11.16</td></tr><tr><td>1st-Order on Logit Diff</td><td>11.15</td></tr><tr><td>PGD on Cross-Entropy Loss</td><td>10.38</td></tr><tr><td>PGD on CW Loss</td><td>10.29</td></tr></table>

Table 3: Running time comparison (in seconds) on a single batch of images.  

<table><tr><td>Algorithm</td><td>MNIST</td><td>CIFAR</td><td>IMAGENET</td></tr><tr><td>CW</td><td>16.02</td><td>234.75</td><td>872.28</td></tr><tr><td>DeepFool</td><td>1.14</td><td>21.26</td><td>44.41</td></tr><tr><td>PGD</td><td>0.87</td><td>33.17</td><td>46.3</td></tr><tr><td>FSGM</td><td>0.11</td><td>0.95</td><td>10.05</td></tr><tr><td>Ours (l2)</td><td>3.01</td><td>51.03</td><td>248.82</td></tr></table>

Same as the PGD baseline listed, MARGINATTACK is run with 50 random starts, and the initialization perturbation range  $u = 0.3$ . The number of moves is 500. The target norm is  $\ell_{\infty}$ .  $b_{n} = 5$  and  $a_{n}$  is set as in Eq. (10). The rest of the configuration is the same as in the previous experiments.

Table 2 lists the success rates of different attacks under 0.3 perturbation level. The baseline algorithms are all fix-perturbation attacks, and their results are excerpted from the challenge white-box attack leaderboard. As can be seen, MARGINATTACK, as the only zero-confidence attack algorithm, has the second best result, which shows that it performs competitively against the state-of-the-art fix-perturbation attacks.

# 4.3 CONVERGENCE

We would like to revisit the convergence plot of the constraint value  $c(\pmb{x})$  and perturbation norm  $d(\pmb{x})$  of as in Fig. 2. We can see that MARGINATTACK converges very quickly. In the example shown in the figure, it is able to converge within 20 moves. Therefore, MARGINATTACK can be greatly accelerated. If margin accuracy is the priority, a large number of moves, e.g. 200 as in our experiment, would help. However, if efficiency is the priory, a small number of moves, e.g. 30, suffices to produce a decent attack.

To further assess the efficiency of MARGINATTACK, Tab. 3 compares the running time (in seconds) of attacking one batch of images, implemented on a single NVIDIA Tesla P100 GPU. The batch size is 200 for MNIST and CIFAR10, and 100 for ImageNet. The settings are the same as stated in section 4.1, except that for a better comparison, the number of iterations of CW is cut down to 200, and PGD and MARGINATTACK runs one random pass, so that all the algorithms have the same iteration/moves. Only the  $\ell_2$  versions of MARGINATTACK are shown because the other versions have similar run times. As shown, running time of MARGINATTACK is much shorter than CW, and is comparable to DeepFool and PGD. CW is significantly slower than the other algorithms because it has to run multiple trials to search for the best Lagrange multiplier. Note that DeepFool and CW enable early stop, but MARGINATTACK does not. Considering MARGINATTACK's fast convergence rate, the running time can be further reduced by early stop.

# 5 CONCLUSION

We have proposed MARGINATTACK, a novel zero-confidence adversarial attack algorithm that is better able to find a smaller perturbation that results in misclassification. Both theoretical and empirical analyses have demonstrated that MARGINATTACK is an efficient, reliable and accurate adversarial attack algorithm, and establishes a new state-of-the-art among zero-confidence attacks. What is more, MARGINATTACK still has room for improvement. So far, only two settings of  $a^{(k)}$  and  $b^{(k)}$  are developed, but MARGINATTACK will work for many other settings, as long as assumption 5 is satisfied. Authors hereby encourage exploring novel and better settings for the MARGINATTACK framework, and promote MARGINATTACK as a new robustness evaluation measure or baseline in the field of adversarial attack and defense.

# REFERENCES

Naveed Akhtar and Ajmal Mian. Threat of adversarial attacks on deep learning in computer vision: A survey. arXiv preprint arXiv:1801.00553, 2018.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In Security and Privacy (SP), 2017 IEEE Symposium on, pp. 39-57. IEEE, 2017.  
Ian Goodfellow. Gradient masking causes CLEVER to overestimate adversarial perturbation size. arXiv preprint arXiv:1804.07870, 2018.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin. Black-box adversarial attacks with limited queries and information. arXiv preprint arXiv:1804.08598, 2018.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016a.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial machine learning at scale. arXiv preprint arXiv:1611.01236, 2016b.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Seyed Mohsen Moosavi Dezfooli, Alhussein Fawzi, and Pascal Frossard. DeepFool: a simple and accurate method to fool deep neural networks. In Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), number EPFL-CONF-218057, 2016.  
Ian Goodfellow Reuben Feinman Fartash Faghri Alexander Matyasko Karen Hambardzumyan Yi-Lin Juang Alexey Kurakin Ryan Sheatsley Abhibhav Garg Yen-Chen Lin Nicolas Papernot, Nicholas Carlini. cleverhans v2.0.0: an adversarial machine learning library. arXiv preprint arXiv:1610.00768, 2017.  
Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. In Security and Privacy (EuroS&P), 2016 IEEE European Symposium on, pp. 372-387. IEEE, 2016.  
Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security, pp. 506-519. ACM, 2017.  
JB Rosen. The gradient projection method for nonlinear programming. part ii. nonlinear constraints. Journal of the Society for Industrial and Applied Mathematics, 9(4):514-532, 1961.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211-252, 2015.  
Jiawei Su, Danilo Vasconcellos Vargas, and Sakurai Kouichi. One pixel attack for fooling deep neural networks. arXiv preprint arXiv:1710.08864, 2017.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.

Tsui-Wei Weng, Huan Zhang, Pin-Yu Chen, Jinfeng Yi, Dong Su, Yupeng Gao, Cho-Jui Hsieh, and Luca Daniel. Evaluating the robustness of neural networks: An extreme value theory approach. arXiv preprint arXiv:1801.10578, 2018.  
Tianhang Zheng, Changyou Chen, and Kui Ren. Distributionally adversarial attack. arXiv preprint arXiv:1808.05537, 2018.
