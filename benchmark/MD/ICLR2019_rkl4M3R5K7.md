# OPTIMAL ATTACKS AGAINST MULTIPLE CLASSIFIERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the problem of designing provably optimal adversarial noise algorithms that induce misclassification in settings where a learner aggregates decisions from multiple classifiers. Given the demonstrated vulnerability of state-of-the-art models to adversarial examples, recent efforts within the field of robust machine learning have focused on the use of ensemble classifiers as a way of boosting the robustness of individual models. In this paper, we design provably optimal attacks against a set of classifiers. We demonstrate how this problem can be framed as finding strategies at equilibrium in a two player, zero sum game between a learner and an adversary and consequently illustrate the need for randomization in adversarial attacks. The main technical challenge we consider is the design of best response oracles that can be implemented in a Multiplicative Weight Updates framework to find equilibrium strategies in the zero-sum game. We develop a series of scalable noise generation algorithms for deep neural networks, and show that it outperforms state-of-the-art attacks on various image classification tasks. Although there are generally no guarantees for deep learning, we show this is a well-principled approach in that it is provably optimal for linear classifiers. The main insight is a geometric characterization of the decision space that reduces the problem of designing best response oracles to minimizing a quadratic function over a set of convex polytopes.

# 1 INTRODUCTION

In this paper, we study adversarial attacks that induce misclassification when a learner has access to multiple classifiers. One of the most pressing concerns within the field of AI has been the well-demonstrated sensitivity of machine learning algorithms to noise and their general instability. Seminal work by (Szegedy et al., 2014) has shown that adversarial attacks that produce small perturbations can cause data points to be misclassified by state-of-the-art models, including neural networks. In order to evaluate classifiers' robustness and improve their training, adversarial attacks have become a central focus in machine learning and security (Moosavi-Dezfooli et al., 2016; Koh & Liang, 2017; Liu et al., 2017; Nguyen et al., 2015).

Adversarial attacks induce misclassification by perturbing data points past the decision boundary of a particular class. In the case of binary linear classifiers, for example, the optimal perturbation is to push points in the direction perpendicular to the separating hyperplane. For non-linear models there is no general characterization of an optimal perturbation, though attacks designed for linear classifiers tend to generalize well to deep neural networks (Moosavi-Dezfooli et al., 2016).

Since a learner may aggregate decisions using multiple classifiers, a recent line of work has focused on designing attacks on an ensemble of different classifiers (Liu et al., 2017; Tramer et al., 2018; Abbasi & Gagné, 2017; He et al., 2017). In particular, this line of work shows that an entire set of state-of-the-art classifiers can be fooled by using an adversarial attack on an ensemble classifier that averages the decisions of the classifiers in that set. Given that attacking an entire set of classifiers is possible, the natural question is then:

What is the most effective approach to design attacks on a set of multiple classifiers?

The main challenge when considering attacks on multiple classifiers is that fooling a single model, or even the ensemble classifier (i.e. the model that classifies a data point by averaging individual predictions), provides no guarantees that the learner will fail to classify correctly. Models may have different decision boundaries, and perturbations that affect one may be ineffective on another. Furthermore, a learner can randomize over classifiers and avoid deterministic attacks (see Figure 1).

![](images/c5d7799bc695ecbab5693b98bb428bfaafd69a9f1bb4ed1cd3237c2ac524147b.jpg)  
Figure 1: Illustration of why randomization is necessary to compute optimal adversarial attacks. In this example using binary linear classifiers, there is a single point that is initially classified correctly by two classifiers  $c_{1}, c_{2}$ , and a fixed noise budget  $\alpha$  in the  $\ell_{2}$  norm. A naive adversary who chooses a noise perturbation deterministically will always fail to trick the learner since she can always select the remaining classifier. An optimal adversarial attack in this scenario consists of randomizing with equal probability amongst both noise vectors.

In this paper, we present a principled approach for attacking a set of classifiers which proves to be highly effective. We show that constructing optimal adversarial attacks against multiple classifiers is equivalent to finding strategies at equilibrium in a zero sum game between a learner and an adversary. It is well known that strategies at equilibrium in a zero sum game can be obtained by applying the celebrated Multiplicative Weights Update framework, given an oracle that computes a best response to a randomized strategy. The main technical challenge we address pertains to the characterization and implementation of such oracles. Our main contributions can be summarized as follows:

- We describe the Noise Synthesis Framework (henceforth NSFW) for generating adversarial attacks. This framework reduces the problem of designing optimal adversarial attacks for a general set of classifiers to constructing a best response oracle in a two player, zero sum game between a learner and an adversary;  
- We show that NSFW is an effective approach for designing adversarial noise that fools neural networks. In particular, applying projected gradient descent on an appropriately chosen loss function as a proxy for a best response oracle achieves performance that significantly improves upon current state-of-the-art attacks (see results in Figure 2);  
- We show that applying projected gradient descent on an appropriately chosen loss function is a well-principled approach. We do so by proving that for linear classifiers such an approach yields an optimal adversarial attack if the equivalent game has a pure Nash equilibrium. This result is shown via a geometric characterization of the decision boundary space which reduces the problem of designing optimal attacks to a convex program;  
- If the game does not have a pure Nash equilibrium, there is an algorithm for finding an optimal adversarial attack for linear classifiers whose runtime is exponential in the number of classifiers. We show that finding an optimal strategy in this case is NP-hard.

Paper organization. Following a discussion on related work, in Section 2 we formulate the problem of designing optimal adversarial noise and show how it can be modeled as finding strategies at equilibrium in a two player, zero sum game. Afterwards, we discuss our approach for finding such strategies using MWU and proxies for best response oracles. In Section 2.1, we justify our approach by proving guarantees for linear classifiers. Lastly, in Section 3, we present our experiments.

Additional related work. The field of adversarial attacks on machine learning classifiers has recently received widespread attention from a variety of perspectives (Carlini & Wagner, 2018; Athalye et al., 2018; Elsayed et al., 2018; Papernot et al., 2016b; Schmidt et al., 2018; Bubeck et al., 2018; Madry et al., 2018). In particular, a significant amount of effort has been devoted to computing adversarial examples that induce misclassification across multiple models (Moosavi-Dezfoolii et al., 2017; Szegedy et al., 2014; Moosavi-Dezfoolii et al., 2016). There has been compelling evidence which empirically demonstrates the effectiveness of ensembles as a way of both generating and defending against adversarial attacks. For example, Tramer et al. (2018) establish the strengths of ensemble training as a defense against adversarial attacks. Conversely, Liu et al. (2017) provide the first set of experiments showing that attacking an ensemble classifier is an effective way of generating adversarial examples that transfer to the underlying models. Relative to their investigation, our work differs in certain key aspects. Rather than analyzing adversarial noise from a security perspective and developing methods for black-box attacks, we approach the problem from a theoretical point of view and introduce a formal characterization of the optimal attack against a set of classifiers. Furthermore, by analyzing noise in the linear setting, we design algorithms for this task that have strong guarantees of performance. Through our experiments, we demonstrate how these algorithms motivate a natural extension for noise in deep learning that achieves state-of-the-art results.

# 2 A FRAMEWORK FOR OPTIMAL ADVERSARIAL ATTACKS

Given a set of point-label pairs  $\{(x_i, y_i)\}_{i=1}^m$  where  $(x_i, y_i) \in \mathbf{R}^d \times [k]$ , a deterministic adversarial attack is a totally ordered set of noise vectors,  $V = (v_1, \ldots, v_m) \in \mathbf{R}^{d \times m}$ . We say that  $\mathbf{q}$  is an adversarial attack if  $\mathbf{q}$  is a distribution over sets of noise vectors. An adversarial attack  $\mathbf{q}$  is  $\alpha$ -bounded if for all sets  $V$  that have non-zero probability under  $\mathbf{q}$ , each individual noise vector  $v_i \in V$  has bounded norm, e.g.  $||v_i||_p \leq \alpha$ . We focus on the case where each vector  $v_i$  is bounded to have  $\ell_2$  norm less than a fixed value  $\alpha$ , however, our model can be easily extended to a variety of norms.

For a given classifier  $c: \mathbf{R}^d \to [k]$ , a realization of the adversarial attack,  $V = (v_1, \ldots, v_m)$ , induces misclassification on  $(x_j, y_j)$  if  $c(x_j + v_j) \neq y_j$ . Given a finite set of classifiers  $\mathcal{C}$  and a data set  $S = \{(x_i, y_i)\}_{i=1}^m$  of point-label pairs as above, an optimal adversarial attack is a distribution  $\mathbf{q}$  over sets of noise vectors that maximizes the minimum 0-1 loss of the classifiers in  $\mathcal{C}$ :

$$
\arg \max  _ {\mathbf {q}} \min  _ {c \in \mathcal {C}} \frac {1}{m} \sum_ {j \in [ m ]} \underset {V \sim \mathbf {q}} {\mathbb {E}} \left[ \ell_ {0 - 1} \left(c, x _ {j} + v _ {j}, y _ {j}\right) \right] \tag {1}
$$

Optimal adversarial attacks are equilibrium strategies in a zero sum game. An equivalent interpretation of the optimization problem described in Equation (1) is that of a best response in a two player, zero sum game played between a learner and an adversary. When the learner plays classifier  $c \in \mathcal{C}$  and the adversary plays an attack  $V$ , the payoff to the adversary is  $M(c, V) = \frac{1}{m} \sum_{j \in [m]} \ell_{0-1}(c, x_j + v_j, y_j)$ , which is the average 0-1 loss of the learner. The learner and the adversary can choose to play randomized strategies  $\mathbf{p}, \mathbf{q}$  over classifiers and noise vectors yielding expected payout  $\mathbb{E}_{(c, V) \sim (\mathbf{p}, \mathbf{q})} M(c, V)$ . The (mixed) equilibrium strategy of the game is the pair of distributions  $\mathbf{p}, \mathbf{q}$  that maximize the minimum loss  $\max_{\mathbf{p}} \mathbb{E}_{(c, V) \sim (\mathbf{p}, \mathbf{q})} M(c, V)$ .

Computing optimal adversarial attacks via MWU. As discussed above, the optimization problem of designing optimal adversarial attacks reduces to that of finding strategies at equilibrium in a zero sum game. It is well known that the celebrated Multiplicative Weight Updates algorithm can be used to efficiently compute equilibrium strategies of zero sum games when equipped with a best response oracle that finds an optimal set of perturbations for any strategy chosen by the learner:

$$
\operatorname {B E S T R E S P O N S E} (\mathbf {p}, \alpha) \stackrel {\text {d e f}} {=} \underset {V \in \mathbf {R} ^ {d \times m}} {\arg \max } \mathbb {E} _ {c \sim \mathbf {P}} [ M (c, V) ]; \quad \mathrm {s . t} \| v _ {i} \| _ {2} \leq \alpha \forall v _ {i} \in V \tag {2}
$$

Our framework for generating adversarial noise applies the Multiplicative Weight Updates algorithm as specified in Algorithm 1. The algorithm returns distributions  $\mathbf{p}^{\star},\mathbf{q}^{\star}$  that are within  $\delta$  of the equilibrium value of the game  $\lambda = \min_{\mathbf{p}}\max_{\mathbf{q}}\mathbb{E}_{(c,V)\sim (\mathbf{p},\mathbf{q})}[M(c,V)]$  by using  $T\in \mathcal{O}(\frac{\ln n}{\delta^2})$  calls to a best response oracle. In this work, we focus on developing attacks on neural networks and linear models. Yet, our framework is general enough to generate optimal attacks for any domain in which one can approximate a best response. We analyze the convergence of NSFW in Appendix G.

Approximating a best response. Given the framework described above, the main challenge is in computing a best response strategy. To do so, at every iteration, as a proxy for a best response, we apply projected gradient descent (PGD) to an appropriately chosen surrogate loss function. In particular, given  $\mathcal{C} = \{c_i\}_{i=1}^n$  for every  $(x,y) \in \mathbf{R}^d \times [k]$  we aim to solve:

$$
\max  _ {v} \sum_ {i = 1} ^ {n} \mathbf {p} [ i ] \ell \left(c _ {i}, x + v, y\right) \tag {3}
$$

$\ell$  is a loss function that depends on the type of attack (targeted vs. untargeted) and the type of classifiers in  $\mathcal{C}$  (linear vs. deep). We introduce a series of alternatives for  $\ell$  in the following section.

As we will now show, maximizing the loss of the learner by applying PGD to a weighted sum of loss functions is a well-principled approach to computing best responses as it is guaranteed to converge to the optimal solution in the case where  $\mathcal{C}$  is composed of linear classifiers. While there are generally no guarantees for solving non-convex optimization problems of this sort for deep neural networks, in Section 3, we demonstrate the effectiveness of our approach by showing that it experimentally improves upon current state-of-the-art attacks.

For example, see Appendix H for extensions to the  $\ell_{\infty}$  norm.  
The adversary plays the role of the max player in our two player, zero sum game.  
In our experiments in Section 3, we show that the algorithm converges in a far fewer number of iterations.

Algorithm 1 Noise Synthesis FrameWork (NSFW)  
Input: Classifiers  $\mathcal{C} = \{c_1,\dots ,c_n\}$  data points  $\{(x_i,y_i)\}_{i = 1}^m$  parameters  $\alpha ,T$  initialize  $\mathbf{p}_1 = (\frac{1}{n},\ldots ,\frac{1}{n})$  .  $\epsilon = \sqrt{\ln|\mathcal{C}| / T}$    
for  $t = 1$  to  $T$  do Set  $V_{t} =$  BEST RESPONSE  $(\mathbf{p}_t,\alpha)$  Set  $\mathbf{p}_{t + 1}[i]\propto \mathbf{p}_t[i](1 - \epsilon)^{M(c_i,V_t)}$  for every  $i\in [n]$    
end for   
Return: uniform distributions  $\mathbf{p}^{\star},\mathbf{q}^{\star}$  over  $\mathbf{p}_1,\dots ,\mathbf{p}_T$  .  $V_{1},\ldots ,V_{T}$

# 2.1 PROVABLE GUARANTEES FOR COMPUTING OPTIMAL NOISE

The main theoretical insight that leads to provable guarantees for generating adversarial noise is a geometric characterization of the underlying structure of adversarial attacks. Regardless of the type of model, selecting a distribution over classifiers partitions the input space into disjoint regions, each of which is associated with a single loss value for the learner. Given a distribution over classifiers played by the learner, computing a best response strategy for the adversary then reduces to a search problem. In this problem, the search is for points in each region that lie within the noise budget and can be misclassified. The best response is to select the region which induces the maximal loss.

In the case of linear classifiers, the key observation is that the regions are convex. As a result, designing optimal adversarial attacks reduces to solving a series of quadratic programs.

Lemma 1. Selecting a distribution  $\mathbf{p}$  over a set  $\mathcal{C}$  of  $n$  linear classifiers, partitions the input space  $\mathbf{R}^d$  into  $k^n$  disjoint, convex sets  $T_j$  such that:

1. For each  $T_j$ , there exists a unique label vector  $s_j \in [k]^n$  such that for all  $x \in T_j$  and  $c_i \in \mathcal{C}$ ,  $c_i(x) = s_{j,i}$ , where  $s_{j,i}$  is a particular label in  $[k]$ .  
2. There exists a finite set of numbers  $a_1, \ldots, a_{k^n}$ , not necessarily all unique, such that  $\sum_{i=1}^{n} \mathbf{p}[i] \ell_{0-I}(c_i, x, y) = a_j$  for a fixed  $y$  and all  $x \in T_j$  
3.  $\mathbf{R}^d\setminus \bigcup_jT_j$  is a set of measure zero.

Proof Sketch (see full proof in Appendix C). Each set  $T_{j}$  is defined according to the predictions of the classifiers  $c_{i} \in \mathcal{C}$  on points  $x \in T_{j}$ . In particular, each region  $T_{j}$  is associated with a unique label vector  $s_{j} \in [k]^{n}$  s.t  $c_{i}(x) = s_{j,i}$  for all  $c_{i} \in \mathcal{C}$ . Since the prediction of each classifier is the same for all points in a particular region, the loss of the learner  $\sum_{i \in [n]} \mathbf{p}[i] \ell_{0-1}(c_{i}, x, y)$  is constant over the entire region. Convexity then follows by showing that each  $T_{j}$  is an intersection of hyperplanes.

This characterization of the underlying geometry now allows us to design best response oracles for linear classifiers via convex optimization. For our analysis, we focus on the case where  $\mathcal{C}$  consists of "one-vs-all" classifiers. In the appendix, we show how our results can be generalized to other methods for multilabel classification by reducing these other approaches to the "one-vs-all" case. Given  $k$  classes, a "one-vs-all" classifier  $c_{i}$  consists of  $k$  linear functions  $c_{i,j}(x) = \langle w_{i,j}, x \rangle + b_{i,j}$  where  $j \in [k]$ . On input  $x$ , predictions are made according to the rule  $c_{i}(x) = \arg \max_{j} c_{i,j}(x)$ .

Lemma 2. For linear classifiers, implementing a best response oracle reduces to the problem of minimizing a quadratic function over a set of  $k^n$  convex polytopes.

Proof Sketch (see full proof in Appendix C). The main idea behind this lemma is that given a distribution over classifiers, the loss of the learner can be maximized individually for each point  $(x,y) \in S$ . Furthermore, by Lemma 1, the loss can assume only finitely many values, each of which is associated with a particular convex region  $T_{j}$  of the input space. Therefore, to compute a best response, we can iterate over all regions and choose the one associated with the highest loss. To find points in each region  $T_{j}$ , we can simply minimize the  $\ell_2$  norm of a perturbation  $v$  such that  $x + v \in T_{j}$ , which can be framed as minimizing a quadratic function over a convex set.

These results give an important characterization, but it also shows that the number of polytopes is exponential in the number of classifiers. To overcome this difficulty, we demonstrate how when there exists a pure strategy Nash equilibrium (PSNE), that is a single set of noise vectors  $V$  where every vector is bounded by  $\alpha$  and  $\min_{c_i \in \mathcal{C}} M(c_i, V) = 1$ , PGD applied to the reverse hinge loss,  $\ell_r$ , is

guaranteed to converge to a point that achieves this maximum for binary classifiers. More generally, given a label vector  $s_j \in [k]^n$ , PGD applied to the targeted reverse hinge loss,  $\ell_t$ , converges to a point within the noise budget that lies within the specified set  $T_j$ . We define  $\ell_r$  and  $\ell_t$  as follows:

$$
\ell_ {r} \left(c _ {i}, x, y\right) \stackrel {\text {d e f}} {=} \left(y \left(\left\langle w _ {i}, x \right\rangle + b _ {i}\right)\right) ^ {+}; \quad \ell_ {t} \left(c _ {i}, x, j\right) \stackrel {\text {d e f}} {=} \left(\max  _ {l \neq j} c _ {i, l} (x) - c _ {i, j}\right) ^ {+} \tag {4}
$$

The proof follows standard arguments for convergence of convex and  $\beta$ -smooth functions.

Theorem 1. Given any precision  $\epsilon >0$  and noise budget  $\alpha >0$ :

- For a finite set of linear binary classifiers  $\mathcal{C}$  and a point  $(x,y)$ , running PGD for  $T = 4\alpha/\epsilon$  iterations on the objective  $f(v) = \sum_{i=i}^{n} \mathbf{p}[i] \ell_r(c_i, x + v, y)$  converges to a point that is within  $\epsilon$  of the pure strategy Nash equilibrium  $f(x + v^*)$ , if such an equilibrium exists;  
- For a finite set of linear multilabel classifiers  $\mathcal{C}$ , given a label vector  $s_j \in [k]^n$  and a distribution  $\mathbf{p}$  over  $\mathcal{C}$ , running PGD for  $T = 4\alpha/\epsilon$  iterations on the objective  $f(v) = \sum_{i=i}^{n} \mathbf{p}[i]\ell_t(c_i, x + v, s_{j,i})$  converges to a point  $x + v^{(T)}$  such that  $f(x + v^{(T)}) - f(x + v^*) \leq \epsilon$  where  $x + v^* \in T_j$  and  $||v^*||_2 \leq \alpha$ , if such a point exists.

Proof Sketch. From the definition of the reverse hinge loss, we see that  $\ell_r(c_i,x',y) = 0$  if and only if  $\ell_{0 - 1}(c_i,x',y) = 1$ . Similarly, the targeted loss  $\ell_t(c_i,x',j)$  is 0 if and only if  $c_{i}$  predicts  $x^{\prime}$  to have label  $j$ . For linear classifiers, both of these functions are convex and  $\beta$ -smooth. Hence PGD converges to a global minimum, which is zero if there exists a pure equilibrium in the game.

The requirement that there exist a feasible point  $x'$  within  $T_{j}$  is not only sufficient, it is also necessary in order to avoid a brute force search. Designing an efficient algorithm to find the region associated with the highest loss is unlikely as the decision version of the problem is NP-hard even for binary linear classifiers. We state the theorem below and defer the proof to the appendix.

Theorem 2. Given a set  $\mathcal{C}$  of  $n$  binary, linear classifiers, a number  $B$ , a point  $(x,y)$ , noise budget  $\alpha$ , and a distribution  $\mathbf{p}$ , finding  $v$  with  $||v||_2 \leq \alpha$  s.t. the loss of the learner is exactly  $B$  is NP-complete.

As we show in the following section, this hardness result does not limit our ability to compute optimal adversarial examples. Most of the problems that have been examined in the context of adversarial noise suppose that the learner has access only to a small number of classifiers (e.g. less than 5) (Liu et al., 2017; Dong et al., 2017; Abbasi & Gagné, 2017; Tramer et al., 2018; He et al., 2017). In such cases we can solve the convex program over all regions and find an optimal adversarial attack, even when a pure Nash equilibrium does not exist.

# 3 EXPERIMENTS

We evaluate the performance of NSFW at fooling a set of classifiers by comparing against noise generated by using state-of-the-art attacks against an ensemble classifier. Recent work by Liu et al. (2017) and Tramer et al. (2018), demonstrates how attacking an ensemble of a set of classifiers generates noise that improves upon all previous attempts at fooling multiple classifiers. We test our methods on deep neural networks on MNIST and ImageNet, as well as on linear classifiers where we know that NSFW is guaranteed to converge to the optimal adversarial attack.

# 3.1 EVALUATING NSFW ON DEEP NEURAL NETWORKS

We use the insights derived from our theoretical analysis of linear models to approximate a best response oracle for this new setting. Specifically, at each iteration of NSFW we compute a best response as in Equation (3) by running PGD on a weighted sum of untargeted reverse hinge losses,  $\ell_{ut}$ , introduced in this domain by Carlini & Wagner (2017). Given a network  $c_{i}$ , we denote  $c_{i,j}(x)$  to be the probability assigned by the model to input  $x$  belonging to class  $j$  (the  $j$ th output of the softmax layer of the model).

$$
\ell_ {u t} \left(c _ {i}, x, y\right) \stackrel {\text {d e f}} {=} \left(c _ {i, y} (x) - \max  _ {j \neq y} c _ {i, j} (x)\right) ^ {+} \tag {5}
$$

For MNIST, the set of classifiers  $\mathcal{C}$  consists of 5 convolutional neural networks, each with a different architecture, that we train on the full training set of  $55\mathrm{k}$  images (see Appendix for details). All classifiers (models) were over  $97\%$  accurate on the MNIST test set. For ImageNet,  $\mathcal{C}$  consists of the InceptionV3, DenseNet121, ResNet50, VGG16, and Xception models with pre-trained weights

![](images/55b589649c971c0ad39d123b8a4a87277c3cad7d9ab7c6ce511c379a53e7e6c2.jpg)

![](images/761b3f7009d38e14e07326e14a58e5a6ba862c50c8636b610485f236e2572eb2.jpg)  
Figure 2: Visual comparison of misclassification using state-of-the-art adversarial attacks. We compare the level of noise necessary to induce similar levels of misclassification by attacking an ensemble classifier using the (from left to right) Fast Gradient Method (FGM), the Madry attack, and the Momentum Iterative Method (MIM) versus applying NSFW (rightmost column) on the same set of classifiers. To induce a maximum of  $17\%$  accuracy across all models, we only need to set  $\alpha$  to be 300 for NSFW. For the MIM attack on the ensemble we need to set  $\alpha = 2000$ . For FGM and the Madry attack, the noise budget must be further increased to 8000.

![](images/d2e9c68a1e3faf7ced663fbcaeafc525182aea443b9851c30a940de86092278c.jpg)

![](images/5fc1e90c09669a28ab1736542c04d977a59ce0ec7a590a2d3eb2b3fbe7957879.jpg)

![](images/befd8c3f0cbb34ca2855973b5d577cd70a34e5df89f069463114e40359da828e.jpg)

![](images/bc9b25c8a1ad111fa7211477edf078239cf1ace7bb41295abf114af3e2f5df11.jpg)

![](images/8eabb994cbdba43e744ff81c8d3aafc0b9794b6d1409e045c1621e322a8b7806.jpg)

![](images/24fc751e61811f7b75492a0026670bb2fb414160567bb9ac80261c49d81153d4.jpg)

downloaded from the Keras repository (Chollet et al., 2015; He et al., 2016; Simonyan & Zisserman, 2014; Chollet, 2017; Szegedy et al., 2016; Huang et al., 2017).<sup>4</sup>

To evaluate the merits of our approach, we compare our results against attacks on the ensemble composed of  $\mathcal{C}$  as suggested by Liu et al. (2017). More specifically, we create an ensemble by averaging the outputs of the softmax layers of the different networks using equal weights. We generate baseline attacks by attacking the ensemble using (1) the Fast Gradient Method by Goodfellow et al. (2014), (2) the Projected Gradient Method by Madry et al. (2018), and (3) the Momentum Iterative Method by Dong et al. (2017) which we download from the Cleverhans library (Papernot et al., 2016a).<sup>5</sup>

We select the noise budget  $\alpha$  by comparing against the average  $\ell_2$  distortion reported by similar papers in the field. For MNIST, we base ourselves off the values reported by Carlini & Wagner (2017) and choose a noise budget of 3.0. For ImageNet, we compare against Liu et al. (2017). In their paper, they run similar untargeted experiments on ImageNet with 100 images and report a noise budget of 22 when measured as the root mean squared deviation. Converted to the  $\ell_2$  norm, this corresponds to  $\alpha \geq 8500$ . We found this noise budget to be excessive, yielding images comparable to those in the leftmost column in Figure 2. Therefore, we chose  $\alpha = 300$  (roughly  $3.5\%$  of the total distortion used in Liu et al. (2017)) which ensures that the perturbed images are visually indistinguishable from the originals to the human eye (see rightmost column in Figure 2).

<table><tr><td>Noise Algorithm</td><td>InceptionV3</td><td>Xception</td><td>ResNet50</td><td>DenseNet121</td><td>VGG16</td><td>Mean</td><td>Max</td></tr><tr><td>FGM</td><td>74%</td><td>77%</td><td>60%</td><td>54%</td><td>66%</td><td>66%</td><td>77%</td></tr><tr><td>Madry Attack</td><td>74%</td><td>76%</td><td>58%</td><td>53%</td><td>73%</td><td>67%</td><td>76%</td></tr><tr><td>Momentum Iterative Method</td><td>68%</td><td>65%</td><td>34%</td><td>35%</td><td>49%</td><td>50%</td><td>68%</td></tr><tr><td>NSFW</td><td>17%</td><td>12.2%</td><td>5.8%</td><td>7.2%</td><td>13.4%</td><td>11%</td><td>17%</td></tr></table>

Table 1: Accuracies of ImageNet models under different noise algorithms using a noise budget of 300.0 in the  $\ell_2$  norm. Entry  $(i,j)$  indicates the accuracy of each model  $j$  when evaluated on noise from attack  $i$ . The last two columns report the mean and max accuracy of the classifiers on a particular attack. We see that NSFW significantly outperforms noise generated by an ensemble classifier for all choices of attack algorithms.

![](images/88d0a8e2d8af9ca6bdbe242850d3782a1ad85ea58ead75cce5c5cff7b211150a.jpg)  
Figure 3: Class saliency map with respect to the image displayed in the top row of Figure 2 for each ImageNet classifier and their ensemble. From left to right: InceptionV3, Xception, ResNet50, DenseNet121, VGG16, and the ensemble classifier of all 5 models.

![](images/9780140182124fbfe548bc65038d50f96772dadd26d7cd37ce481be8c1ad2d21.jpg)

![](images/090e3d432be60acbcc611d8ef28d4bd5f0bde9243ec29b531759e93e6c7817f5.jpg)

![](images/ffb9678bb61cc514d05b1e4a7e26a2c25ae4c189108c3f99a19ca6b59f0a1e21.jpg)

![](images/bae3ea234580c888cbc539447fd66f4486592e07bda830da8d2a1050a17867dd.jpg)

![](images/39e22a4edea824659c4f1fb50d5cc21bc651c6431caaa76f68a100041cdc2bc7.jpg)

For our experiments, we ran NSFW for 50 MWU iterations on MNIST models and for 10 iterations on ImageNet classifiers. We use far fewer iterations than the theoretical bound since we found that in practice NSFW converges to the equilibrium solution in only a small number of iterations (see Figure 5 in Appendix A). At each iteration of the MWU we approximate a best response as described in Equation 3 by running PGD using the Adam optimizer (Kingma & Ba, 2014) on a sum of untargeted reverse hinge losses. Specifically, we run the optimizer for 5k iterations with a learning rate of .01. At each iteration, we clip images to lie in the range [0, 1] for MNIST and [0, 255.0] for ImageNet.<sup>7</sup>

Finally, for evaluation, for both MNIST and ImageNet we selected 100 images uniformly at random from the set of images in the test sets that were correctly classified by all models. In Table 1, we report the empirical accuracy of all classifiers in the set  $\mathcal{C}$  when evaluated on NSFW as well as on the three baseline attacks. To compare their performance, we highlight the average and maximum accuracies of models in  $\mathcal{C}$  when attacked using a particular noise solution.

From Table 1, we see that on ImageNet our algorithm results in solutions that robustly optimize over the entire set of models using only a small amount of noise. The maximum accuracy of any classifier is  $17\%$  under NSFW, while the best ensemble attack yields a max accuracy of only  $68\%$ . If we wish to generate a similar level of performance from the ensemble baselines, we would need to increase the noise budget to 8000 for FGM and the Madry attack and to 2000 for the Momentum Iterative Method. We present a visual comparison of the different attacks under these noise budgets required to achieve accuracy of  $17\%$  in Figure 2. On MNIST, we find similar results. NSFW yields a max accuracy of  $22.6\%$  compared to the next best result of  $48\%$  generated by the Madry attack on the ensemble. We summarize the results for MNIST in Table 2 presented in Appendix A.

# 3.2 WHY ARE DIRECT ATTACKS ON ENSEMBLE NETWORKS POOR NOISE GENERATORS?

ANALYZING DIFFERENCES IN DECISION BOUNDARIES VIA CLASS SALIENCY MAPS

As seen in the previous section, noise generated by directly attacking an ensemble of classifiers significantly underperforms NSFW at robustly fooling the underlying models. In this section, we aim to understand this phenomenon by analyzing how the decision boundary of the ensemble model compares to that of the different networks. In particular, we visualize the class boundaries of convolutional neural networks using the algorithm proposed by Simonyan et al. (2013) for generating saliency maps. $^{8}$  The class saliency map indicates which features (pixels) are most relevant in classifying an image to have a particular label. $^{9}$  Therefore, they serve as one way of understanding the decision boundary of a particular model by highlighting which dimensions carry the highest weight.

In Figure 3, we see that the class saliency maps for individual models exhibit significant diversity. The ensemble of all 5 classifiers appears to contain information from all models, however, certain regions that are of central importance for individual models are relatively less prominent in the ensemble saliency map. Compared to our approach which calculates individual gradients for classifiers in  $\mathcal{C}$ , creating an ensemble classifier obfuscates key information regarding the decision boundary of individual models. We make this discussion rigorous by analyzing the linear case in Appendix B.

![](images/e1162118f22fcaf6c6497337417a310084f1150586407c68d505f1c5e34d2498.jpg)  
Figure 4: Results of running NSFW on linear models. On the left, we demonstrate the results of running NSFW on linear multiclass models using different noise functions and varying the noise budget  $\alpha$ . NSFW-Oracle corresponds to running Algorithm 1 using the best response oracle described in Lemma 2. Similarly, NSFW-Untargeted shows the results of running NSFW and applying PGD to a weighted sum of untargeted losses as in Equation (3). The label iteration method is described below. Lastly, the ensemble attack corresponds to the optimal noise on an equal weights ensemble of models in  $\mathcal{C}$ . On the right, we illustrate the convergence of NSFW on linear binary classifiers with maximally different decision boundaries to compare against the convergence rate observed for neural nets in Figure 5 and better understand when weight adaptivity is necessary.

![](images/88837121124cf597d3954609ac58fbedcbe4fb908f8d8223cff57d501a38cf52.jpg)

# 3.3 EXPERIMENTS ON LINEAR CLASSIFIERS

In addition to evaluating our approach on neural networks, we performed experiments with linear classifiers. Since we have a precise characterization of the optimal attack on a set of linear classifiers, we can rigorously analyze the performance of different methods in comparison to the optimum.

We train two sets of 5 linear SVM classifiers on MNIST, one for binary classification (digits 4 and 9) and another for multiclass (first 4 classes, MNIST 0-3). To ensure a diversity of models, we randomly zero out up to  $75\%$  of the dimensions of the training set for each classifier. Hence, each model operates on a random subset of features. All models achieve test accuracies of above  $90\%$ . For our experiments, we select 1k points from each dataset that are correctly classified by all models.

In order to better compare across different best response proxies, we further extend NSFW by incorporating the label iteration method as another heuristic to generate untargeted noise. Given a point  $(x,y)$ , the iterative label method attempts to calculate a best response by running PGD on the targeted reverse hinge loss for every label  $j\in [k]\setminus \{y\}$  and choosing the attack associated with the minimal loss. Compared to the untargeted reverse hinge loss, it has the benefit of being convex.

As for deep learning classifiers, we compare our results to the noise generated by the optimal attack on an ensemble of models in  $\mathcal{C}$ . Since the class of linear classifiers is convex, creating an equal weights ensemble by averaging the weight vectors results in just another linear classifier. We can compute the optimal attack by running the best response oracle described in Section 2.1 for the special case where  $\mathcal{C}$  consists of a single model and then scaling the noise to have norm equal to  $\alpha$ .

As seen in the leftmost plot in Figure 4, even for linear models there is a significant difference between the optimal attack and other approaches. Specifically, we observe an empirical gap between NSFW equipped with the best response oracle as described in Lemma 2 vs. NSFW with proxy best response oracles, e.g. the oracle that runs PGD on appropriately chosen loss functions. $^{10}$  This difference in performance is consistent across a variety of noise budgets. Our main takeaway is that in theory and in practice, there is a significant benefit in applying appropriately designed best response oracles. Lastly, on the right in Figure 4, we illustrate how the adaptivity of MWU is in general necessary to compute optimal attacks. While for most cases, NSFW converges to the equilibrium solution almost immediately, if the set of classifiers is sufficiently diverse, running NSFW for a larger number of rounds drastically boosts the quality of the attack. (See Appendix A for details.)

# 4 CONCLUSION

Designing adversarial attacks when a learner has access to multiple classifiers is a non-trivial problem. In this paper we introduced NSFW which is a principled approach that is provably optimal on linear classifiers and empirically effective on neural networks. The main technical crux is in designing best response oracles which we achieve through a geometrical characterization of the optimization landscape. We believe NSFW can generalize to domains beyond those in this paper.

# REFERENCES

Mahdieh Abbasi and Christian Gagné. Robustness to adversarial examples through an ensemble of specialists. CoRR, abs/1702.06856, 2017. URL http://arxiv.org/abs/1702.06856.  
Anish Athalye, Nicholas Carlini, and David A. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, July 10-15, 2018, pp. 274-283, 2018. URL http://proceedings.mlr.press/v80/athalye18a.html.  
Sebastien Bubeck. Convex optimization: Algorithms and complexity. Foundations and Trends in Machine Learning, 8(3-4):270, November 2015. ISSN 1935-8237. doi: 10.1561/2200000050. URL http://dx.doi.org/10.1561/2200000050.  
Sebastien Bubeck, Eric Price, and Ilya P. Razenshteyn. Adversarial examples from computational constraints. CoRR, abs/1805.10204, 2018.  
Nicholas Carlini and David A. Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy, SP 2017, San Jose, CA, USA, May 22-26, 2017, pp. 39-57, 2017. doi: 10.1109/SP.2017.49.  
Nicholas Carlini and David A. Wagner. Audio adversarial examples: Targeted attacks on speech-to-text. In 2018 IEEE Security and Privacy Workshops, SP Workshops 2018, San Francisco, CA, USA, May 24, 2018, pp. 1-7, 2018. doi: 10.1109/SPW.2018.00009. URL https://doi.org/10.1109/SPW.2018.00009.  
François Chollet et al. Keras. https://keras.io, 2015.  
Franois Chollet. Xception: Deep learning with depthwise separable convolutions. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1800-1807, 2017.  
Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Xiaolin Hu, and Jun Zhu. Discovering adversarial examples with momentum. CoRR, abs/1710.06081, 2017. URL http://arxiv.org/abs/1710.06081.  
Gamaleldin Fathy Elsayed, Shreya Shankar, Brian Cheung, Nicolas Papernot, Alex Kurakin, Ian Goodfellow, and Jascha Sohl-dickstein. Adversarial examples that fool both computer vision and time-limited human. 2018. URL https://arxiv.org/pdf/1802.08195.pdf.  
Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. J. Comput. Syst. Sci., 55(1):119-139, August 1997. ISSN 0022-0000. doi: 10.1006/jcss.1997.1504. URL http://dx.doi.org/10.1006/jcss.1997.1504.  
Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. CoRR, abs/1412.6572, 2014. URL http://arxiv.org/abs/1412.6572.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.  
Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song. Adversarial example defense: Ensembles of weak defenses are not strong. In 11th USENIX Workshop on Offensive Technologies, WOOT 2017, Vancouver, BC, Canada, August 14-15, 2017., 2017. URL https://www.usenix.org/conference/woot17/workshop-program/presentation/he.  
Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2261-2269, 2017.  
Satyen Kale. Efficient algorithms using the multiplicative weights update method, January 2007. URL http://search.proquest.com/docview/304824121/.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.  
Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, pp. 1885-1894, 2017. URL http://proceedings.mlr.press/v70/koh17a.html.  
Alexey Kurakin, Ian J. Goodfellow, Samy Bengio, Yinpeng Dong, Fangzhou Liao, Ming Liang, Tianyu Pang, Jun Zhu, Xiaolin Hu, Cihang Xie, Jianyu Wang, Zhishuai Zhang, Zhou Ren, Alan L. Yuille, Sangxia Huang, Yao Zhao, Yuzhe Zhao, Zhonglin Han, Junjiajia Long, Yerkebulan Berdibekov, Takuya Akiba, Seiya Tokui, and Motoki Abe. Adversarial attacks and defences competition. CoRR, abs/1804.00097, 2018. URL http://arxiv.org/abs/1804.00097.  
Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delving into transferable adversarial examples and black-box attacks. In Proceedings of 5th International Conference on Learning Representations, 2017.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJzIBfZAb.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: A simple and accurate method to fool deep neural networks. In CVPR, pp. 2574-2582. IEEE Computer Society, 2016.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 86-94, 2017.  
Anh Mai Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pp. 427-436, 2015. doi: 10.1109/CVPR.2015.7298640. URL https://doi.org/10.1109/CVPR.2015.7298640.  
Nicolas Papernot, Ian Goodfellow, Ryan Sheatsley, Reuben Feinman, and Patrick McDaniel. cleverhans v1.0.0: an adversarial machine learning library. arXiv preprint arXiv:1610.00768, 2016a.  
Nicolas Papernot, Patrick D. McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, and Ananthram Swami. The limitations of deep learning in adversarial settings. 2016 IEEE European Symposium on Security and Privacy, pp. 372-387, 2016b.  
Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, and Aleksander Madry. Adversarily robust generalization requires more data. CoRR, abs/1804.11285, 2018.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. URL http://arxiv.org/abs/1409.1556.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. CoRR, abs/1312.6034, 2013. URL http://arxiv.org/abs/1312.6034.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014. URL http://arxiv.org/abs/1312.6199.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2818-2826, 2016.  
Florian Tramer, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. Ensemble adversarial training: Attacks and defenses. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id= rkZvSe-RZ.
