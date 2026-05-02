# TOWARDS UNDERSTANDING THE DATA DEPENDENCY OF MIXUP-STYLE TRAINING

Anonymous authors

Paper under double-blind review

# ABSTRACT

In the Mixup training paradigm, a model is trained using convex combinations of data points and their associated labels. Despite seeing very few true data points during training, models trained using Mixup seem to still minimize the original empirical risk and exhibit better generalization and robustness on various tasks when compared to standard training. In this paper, we investigate how these benefits of Mixup training rely on properties of the data in the context of classification. For minimizing the original empirical risk, we compute a closed form for the Mixup-optimal classification, which allows us to construct a simple dataset on which minimizing the Mixup loss can provably lead to learning a classifier that does not minimize the empirical loss on the data. On the other hand, we also give sufficient conditions for Mixup training to also minimize the original empirical risk. For generalization, we characterize the margin of a Mixup classifier, and use this to understand why the decision boundary of a Mixup classifier can adapt better to the full structure of the training data when compared to standard training. In contrast, we also show that, for a large class of linear models and linearly separable datasets, Mixup training leads to learning the same classifier as standard training.

# 1 INTRODUCTION

Mixup (Zhang et al., 2018) is a modification to the standard supervised learning setup which involves training on convex combinations of pairs of data points and their labels instead of the original data itself. In the original paper, Zhang et al. (2018) demonstrated that training deep neural networks using Mixup leads to better generalization performance, as well as greater robustness to adversarial attacks and label noise on image classification tasks. The empirical advantages of Mixup training have been affirmed by several follow-up works (He et al., 2019; Thulasidasan et al., 2019; Lamb et al., 2019; Arazo et al., 2019). The idea of Mixup has also been extended beyond the supervised learning setting, and been applied to semi-supervised learning (Berthelot et al., 2019; Sohn et al., 2020), privacy-preserving learning (Huang et al., 2021), and learning with fairness constraints (Chuang & Mroueh, 2021).

However, from a theoretical perspective, Mixup training is still mysterious even in the basic multiclass classification setting – why should the output of a linear mixture of two training samples be the same linear mixture of their labels, especially when considering highly nonlinear models? Despite several recent theoretical results (Guo et al., 2019; Carratino et al., 2020; Zhang et al., 2020; 2021), there is still not a complete understanding of why Mixup training actually works in practice. In this paper, we try to understand why Mixup works by first understanding when Mixup works: in particular, how the properties of Mixup training rely on the structure of the training data.

We consider two properties for classifiers trained with Mixup. First, even though Mixup training does not observe many original data points during training, it usually can still correctly classify all of the original data points (empirical risk minimization (ERM)). Second, the aforementioned empirical works have shown how classifiers trained with Mixup often have better adversarial robustness and generalization than standard training. In this work, we show that both of these properties can rely heavily on the data used for training, and that they need not hold in general.

Main Contributions and Related Work. The idea that Mixup can potentially fail to minimize the original risk is not new; Guo et al. (2019) provide examples of how Mixup labels can conflict with actual data point labels. However, their theoretical results do not characterize the data and

model conditions under which this failure can provably happen when minimizing the Mixup loss. In Section 2 of this work, we provide a concrete classification dataset on which continuous approximate-minimizers of the Mixup loss can fail to minimize the empirical risk. We also provide sufficient conditions for Mixup to minimize the original risk, and show that these conditions hold approximately on standard image classification benchmarks.

With regards to generalization and robustness, the parallel works of Carratino et al. (2020) and Zhang et al. (2020) showed that Mixup training can be viewed as minimizing the empirical loss along with a data-dependent regularization term. Zhang et al. (2020) further relate this term to the adversarial robustness and Rademacher complexity of certain function classes learned with Mixup. In Section 3, we take an alternative approach to understanding generalization and robustness by analyzing the margin of Mixup classifiers. Our perspective can be viewed as complementary to that of the aforementioned works, as we directly consider the properties exhibited by a Mixup-optimal classifier instead of considering what properties are encouraged by the regularization effects of the Mixup loss. In addition to our margin analysis, we also show that for the common setting of linear models trained on high-dimensional Gaussian features both Mixup (for a large class of mixing distributions) and ERM with gradient descent learn the same classifier with high probability.

Finally, we note the related works that are beyond the scope of our paper; namely the many Mixup-like training procedures such as Manifold Mixup (Verma et al., 2019), Cut Mix (Yun et al., 2019), Puzzle Mix (Kim et al., 2020), and Co-Mixup (Kim et al., 2021).

# 2 MIXUP AND EMPIRICAL RISK MINIMIZATION

The goal of this section is to understand when Mixup training can also minimize the empirical risk. Our main technique for doing so is to derive a closed-form for the Mixup-optimal classifier over a sufficiently powerful function class, which we do in Section 2.2 after introducing the basic setup in Section 2.1. We use this closed form to motivate a concrete example on which Mixup training does not minimize the empirical risk in Section 2.3, and show under mild nondegeneracy conditions that Mixup will minimize the empirical risk in Section 2.4.

# 2.1 SETUP

We consider the problem of  $k$ -class classification where the classes  $1, \ldots, k$  correspond to compact disjoint sets  $X_1, \ldots, X_k \subset \mathbb{R}^n$  with an associated probability measure  $\mathbb{P}_X$  supported on  $X = \bigcup_{i=1}^{k} X_i$ . We use  $\mathcal{C}$  to denote the set of all functions  $g: \mathbb{R}^n \to [0,1]^k$  satisfying the property that  $\sum_{i=1}^{k} g^i(x) = 1$  for all  $x$  (where  $g^i$  represents the  $i$ -th coordinate function of  $g$ ). We refer to a function  $g \in \mathcal{C}$  as a classifier, and say that  $g$  classifies  $x$  as class  $j$  if  $j = \operatorname{argmax}_i g^i(x)$ . The cross-entropy loss associated with such a classifier  $g$  is then:

$$
J (g, \mathbb {P} _ {X}) = - \sum_ {i = 1} ^ {k} \int_ {X _ {i}} \log g ^ {i} (x) d \mathbb {P} _ {X} (x)
$$

The goal of standard training is to learn a classifier  $h \in \operatorname{argmin}_{g \in \mathcal{C}} J(g, \mathbb{P}_X)$ . Any such classifier  $h$  will necessarily satisfy  $h^i(x) = 1$  on  $X_i$  since the  $X_i$  are disjoint.

Mixup. In the Mixup version of our setup, we are interested in minimizing the cross-entropy of convex combinations of the original data and their classes. These convex combinations are determined according to a probability measure  $\mathbb{P}_f$  whose support is  $[0,1]$ , and we assume this measure has a density  $f$ . For two points  $s,t\in \bigcup_{i = 1}^{k}X_{i}$ , we let  $z_{st}(\lambda) = \lambda s + (1 - \lambda)t$  (and use  $z_{st}$  when  $\lambda$  is understood) and define the Mixup cross-entropy on  $s,t$  with respect to a classifier  $g$  as:

$$
\ell_ {m i x} (g, s, t, \lambda) = \left\{ \begin{array}{l l} - \log g ^ {i} (z _ {s t}) & s, t \in X _ {i} \\ - \left(\lambda \log g ^ {i} (z _ {s t}) + (1 - \lambda) \log g ^ {j} (z _ {s t})\right) & s \in X _ {i}, t \in X _ {j} \end{array} \right.
$$

Having defined  $\ell_{mix}$  as above, we may write the component of the full Mixup cross-entropy loss corresponding to mixing points from classes  $i$  and  $j$  as:

$$
J _ {m i x} ^ {i, j} (g, \mathbb {P} _ {X}, \mathbb {P} _ {f}) = \int_ {X _ {i} \times X _ {j} \times [ 0, 1 ]} \ell_ {m i x} (g, s, t, \lambda) d (\mathbb {P} _ {X} \times \mathbb {P} _ {X} \times \mathbb {P} _ {f}) (s, t, \lambda)
$$

We omit some, or all of, the arguments of  $J_{mix}^{i,j}$  when they are clear from context. The final Mixup cross-entropy loss is then the sum of  $J_{mix}^{i,j}$  over all  $i,j\in \{1,\dots,k\}$  (corresponding to all possible mixings between classes, including themselves):

$$
J _ {m i x} (g, \mathbb {P} _ {X}, \mathbb {P} _ {f}) = \sum_ {i = 1} ^ {k} J _ {m i x} ^ {i, i} + 2 \sum_ {i = 1} ^ {k} \sum_ {j = i + 1} ^ {k} J _ {m i x} ^ {i, j}
$$

Where the coefficient of 2 in front of the second term comes from the fact that  $J_{mix}^{i,j} = J_{mix}^{j,i}$  from Fubini's Theorem (we consider only classifiers  $g$  for which the  $\ell_{mix}$  terms are defined and integrable).

Relation to Prior Work. We have opted for a more general definition of the Mixup loss (at least when constrained to multi-class classification) than prior works. This is not generality for generality's sake, but rather because many of our results apply to any mixing distribution supported on  $[0,1]$ . One obtains the original Mixup formulation of Zhang et al. (2018) for multi-class classification on a finite dataset by taking the  $X_{i}$  to be finite sets, and choosing  $\mathbb{P}_X$  to be the normalized counting measure (corresponding to a discrete uniform distribution). Additionally,  $\mathbb{P}_f$  is chosen to have density  $\mathrm{Beta}(\alpha ,\alpha)$ , where  $\alpha$  is a hyperparameter.

# 2.2 MIXUP-OPTIMAL CLASSIFIER

Given our setup, we now wish to characterize the behavior of a Mixup-optimal classifier at a point  $x \in \mathbb{R}^n$ . However, if the optimization of  $J_{mix}$  is considered over the class of functions  $\mathcal{C}$ , this is intractable (to the best of our knowledge) due to the lack of regularity conditions imposed on functions in  $\mathcal{C}$ . We thus wish to constrain the optimization of  $J_{mix}$  to a class of functions that is sufficiently powerful (so as to include almost all practical settings) while still allowing for local analysis. To do so, we will need the following definitions, which will also be referenced throughout the results in this section and the next:

$$
A _ {x, \epsilon} ^ {i, j} = \left\{\left(s, t, \lambda\right) \in X _ {i} \times X _ {j} \times [ 0, 1 ]: \lambda s + (1 - \lambda) t \in B _ {\epsilon} (x) \right\}
$$

$$
A _ {x, \epsilon , \delta} ^ {i, j} = \left\{\left(s, t, \lambda\right) \in X _ {i} \times X _ {j} \times [ 0, 1 - \delta ]: \lambda s + (1 - \lambda) t \in B _ {\epsilon} (x) \right\}
$$

$$
X _ {m i x} = \left\{x \in \mathbb {R} ^ {n}: \bigcup_ {i, j} A _ {x, \epsilon} ^ {i, j} \text {h a s p o s i t i v e m e a s u r e f o r e v e r y} \epsilon > 0 \right\}
$$

$$
\xi_ {x, \epsilon} ^ {i, j} = \int_ {A _ {x, \epsilon} ^ {i, j}} d (\mathbb {P} _ {X} \times \mathbb {P} _ {X} \times \mathbb {P} _ {f}) (s, t, \lambda)
$$

$$
\xi_ {x, \epsilon , \lambda} ^ {i, j} = \int_ {A _ {x} ^ {i, j}} \lambda d (\mathbb {P} _ {X} \times \mathbb {P} _ {X} \times \mathbb {P} _ {f}) (s, t, \lambda)
$$

The set  $A_{x,\epsilon}^{i,j}$  represents all points in  $X_i \times X_j$  that have lines between them intersecting an  $\epsilon$ -neighborhood of  $x$ , while the set  $A_{x,\epsilon,\delta}^{i,j}$  represents the restriction of  $A_{x,\epsilon}^{i,j}$  to only those points whose connecting line segments intersect an  $\epsilon$ -neighborhood of  $x$  with  $\lambda$  values bounded by  $1 - \delta$ . The set  $X_{mix}$  corresponds to all points for which every neighborhood factors into  $J_{mix}$ . The  $\xi_{x,\epsilon}^{i,j}$  term represents the measure of the set  $A_{x,\epsilon}^{i,j}$  while  $\xi_{x,\epsilon,\lambda}^{i,j}$  represents the expectation of  $\lambda$  over the same set. To provide better intuition for these definitions, we provide visualizations in Section B of the appendix. We can now define the subset of  $\mathcal{C}$  to which we will constrain our optimization of  $J_{mix}$ .

Definition 2.1. Let  $\mathcal{C}^*$  to be the subset of  $\mathcal{C}$  for which every  $h\in \mathcal{C}^*$  satisfies  $h(x) = \lim_{\epsilon \to 0}\mathrm{argmin}_{\theta \in [0,1]^k}J_{mix}(\theta)|_{B_\epsilon (x)}$  for all  $x\in X_{mix}$  when the limit exists. Here  $J_{mix}(\theta)|_{B_\epsilon (x)}$  represents the Mixup loss for a constant function with value  $\theta$  with the restriction of each term in  $J_{mix}$  to the set  $A_{x,\epsilon}^{i,j}$ .

We immediately justify this definition with the following proposition.

Proposition 2.2. Any function  $h \in \operatorname{argmin}_{g \in \mathcal{C}^*} J_{mix}(g, \mathbb{P}_X, \mathbb{P}_f)$  satisfies  $J_{mix}(h) \leq J_{mix}(g)$  for any continuous  $g \in \mathcal{C}$ .

Proof Sketch. We can argue directly from definitions by considering points in  $X_{\text{mix}}$  for which  $h$  and  $g$  differ.

Proposition 2.2 demonstrates that optimizing over  $\mathcal{C}^*$  is at least as good as optimizing over the subset of  $\mathcal{C}$  consisting of continuous functions, so we cover most cases of practical interest (i.e. optimizing deep neural networks). As such, the term "Mixup-optimal" is intended to mean optimal with respect to  $\mathcal{C}^*$  throughout the rest of the paper. We may now characterize the classification of a Mixup-optimal classifier on  $X_{mix}$ .

Lemma 2.3. For any point  $x \in X_{\text{mix}}$  and  $\epsilon > 0$ , there exists a continuous function  $h_\epsilon$  satisfying:

$$
h _ {\epsilon} ^ {i} (x) = \frac {\xi_ {x , \epsilon} ^ {i , i} + 2 \left(\sum_ {j = i + 1} ^ {k} \xi_ {x , \epsilon , \lambda} ^ {i , j} + \sum_ {j = 1} ^ {i - 1} \left(\xi_ {x , \epsilon} ^ {j , i} - \xi_ {x , \epsilon , \lambda} ^ {j , i}\right)\right)}{\sum_ {q = 1} ^ {k} \xi_ {x , \epsilon} ^ {q , q} + 2 \left(\sum_ {j = q + 1} ^ {k} \xi_ {x , \epsilon , \lambda} ^ {q , j} + \sum_ {j = 1} ^ {q - 1} \left(\xi_ {x , \epsilon} ^ {j , q} - \xi_ {x , \epsilon , \lambda} ^ {j , q}\right)\right)} \tag {1}
$$

With the property that  $h_{\epsilon}(x) \to h(x)$  for every  $h \in \operatorname{argmin}_{g \in \mathcal{C}^*} J_{mix}(g, \mathbb{P}_X, \mathbb{P}_f)$  when the limit exists.

Proof Sketch. We define  $h_\epsilon$  to be  $\left. \operatorname*{argmin}_{\theta \in [0,1]^k} J_{mix}(\theta) \right|_{B_\epsilon(x)}$  and show that this is well-defined and continuous using the strict convexity of the minimization problem.

Remark 2.4. For the important case of finite datasets, it will be shown that the limit above always exists as part of the proof of Theorem 3.2.

Although the expression for  $h_{\epsilon}^{i}$  looks complicated, it just represents the expected location of the point  $x$  on all lines between class  $i$  and other classes, normalized by the sum of the expected locations for all classes. Importantly, we note that while  $h_{\epsilon}$  as defined in Lemma 2.3 is continuous for every  $\epsilon > 0$ , its pointwise limit  $h$  need not be, which we demonstrate below.

Proposition 2.5. Let  $X_{1} = \{(0,1), (0, -1)\}$  and let  $X_{2} = \{(1,0), (-1,0)\}$ , with  $\mathbb{P}_X$  being discrete uniform over  $X_{1} \cup X_{2}$  and  $\mathbb{P}_f$  being continuous uniform over  $[0,1]$ . Then the Mixup-optimal classifier  $h$  is discontinuous at  $(0,0)$ .

Proof Sketch. One may explicitly compute for  $x = (0,0)$  that  $h^1(x) = h^2(x) = \frac{1}{2}$ .

Proposition 2.5 illustrates our first significant difference between Mixup training and standard training: there always exists a minimizer of the empirical cross-entropy  $J$  that can be extended to a continuous function (since a minimizer is constant on the class supports and not constrained elsewhere), whereas depending on the data the minimizer of  $J_{mix}$  can be discontinuous.

# 2.3 A MIXUP FAILURE CASE

With that in mind, several model classes popular in practical applications consist of continuous functions. For example, neural networks with ReLU activations are continuous, and several works have noted that they are Lipschitz continuous with shallow networks having approximately small Lipschitz constant (Scaman & Virmaux, 2019; Fazlyab et al., 2019; Latorre et al., 2020). Given the regularity of such models, we are motivated to consider the continuous approximations  $h_{\epsilon}$  in Lemma 2.3 and see if it is possible to construct a dataset on which  $h_{\epsilon}$  (for a fixed  $\epsilon$ ) can fail to classify the original points correctly. We thus consider the following dataset:

Definition 2.6. [3-Point Alternating Line] We define  $\mathcal{X}_3^2$  to be the binary classification dataset consisting of the points  $\{0,1,2\}$  classified as  $\{1,2,1\}$ . In our setup, this corresponds to  $X_{1} = \{0,2\}$  and  $X_{2} = \{1\}$  with  $\mathbb{P}_X = \frac{1}{3} 1_{\{0,1,2\}}$ .

Intuitively, the reason why Mixup can fail on  $\mathcal{X}_3^2$  is that, for choices of  $\mathbb{P}_f$  that concentrate about  $\frac{1}{2}$ , we will have by Lemma 2.3 that the Mixup-optimal classification in a neighborhood of point 1 should skew towards class 1 instead of class 2 due to the sandwiching of point 1 between points 0 and 2. The canonical choice of  $\mathbb{P}_f$  corresponding to a mixing density of  $\mathrm{Beta}(\alpha, \alpha)$  is one such choice:

Theorem 2.7. Let  $\mathbb{P}_f$  have associated density  $\mathrm{Beta}(\alpha, \alpha)$ . Then for any classifier  $h_\epsilon$  on  $\mathcal{X}_3^2$  (as defined in Lemma 2.3), we may choose  $\alpha$  such that  $h_\epsilon$  does not achieve 0 classification error on  $\mathcal{X}_3^2$ .

Proof Sketch. For any  $\epsilon > 0$ , we can bound the  $\xi$  terms in Equation 1 using the fact that  $\mathrm{Beta}(\alpha, \alpha)$  is strictly subgaussian (Marchal & Arbel, 2017), and then choose  $\alpha$  appropriately.

![](images/5fe90fc97390c437a19c8fb7330ae427e6b34f81ee2d7fb08a58309380462061.jpg)  
(a)  $\alpha = 1$

![](images/3e6900dbfc382b2c011c3fee5a38f28c61a3afddda902d1e30021f38fc5cf896.jpg)  
Figure 1: Training error for Mixup and regular training on  $\mathcal{X}_3^2$ . Each curve corresponds to the mean of 10 training runs, and the area around each curve represents a region of one standard deviation.  
(b)  $\alpha = 32$

![](images/c276c08d4e4e8a939ee0d914a7b25cc531a26548505ed3877e59f1a93fa53826.jpg)  
(c)  $\alpha = 128$

Experiments. The result of Theorem 2.7 leads us to believe that the Mixup training of a continuous model should fail on  $\mathcal{X}_3^2$  for appropriately chosen  $\alpha$ . To verify that the theory predicts the experiments, we train a two-layer feedforward neural network with 512 hidden units and ReLU activations on  $\mathcal{X}_3^2$  with and without Mixup. The implementation of Mixup training does not differ from the theoretical setup; we uniformly sample pairs of data points and train on their mixtures. Our implementation uses PyTorch (Paszke et al., 2019) and is based heavily on the open source implementation of Manifold Mixup (Verma et al., 2019) by Shivam Saboo. Results for training using (full-batch) Adam (Kingma & Ba, 2015) with the suggested (and common) hyperparameters of  $\beta_{1} = 0.9, \beta_{2} = 0.999$  and a learning rate of 0.001 are shown in Figure 1. The class 1 probabilities for each point in the dataset outputted by the learned Mixup classifiers from Figure 1 are shown in Table 1 below:

Table 1: Mixup model evaluations on  $\mathcal{X}_3^2$  for different choices of  $\alpha$ .  

<table><tr><td>h</td><td>0</td><td>1</td><td>2</td></tr><tr><td>α = 1</td><td>0.996</td><td>0.174</td><td>0.977</td></tr><tr><td>α = 32</td><td>0.999</td><td>0.604</td><td>0.994</td></tr><tr><td>α = 128</td><td>1.000</td><td>0.654</td><td>0.995</td></tr></table>

We see from Figure 1 and Table 1 that Mixup training fails to correctly classify the points in  $\mathcal{X}_3^2$  for  $\alpha = 32$ , and this misclassification becomes more exacerbated as we increase  $\alpha$ . The choice of  $\alpha$  for which misclassifications begin to happen is largely superficial; we show in Section D of the Appendix that it is straightforward to construct datasets in the style of  $\mathcal{X}_3^2$  for which Mixup training will fail even for the very mild choice of  $\alpha = 1$ . We focus on the case of  $\mathcal{X}_3^2$  here to simplify the theory. The key takeaway is that, for datasets that exhibit (approximately) collinear structure amongst points, it is possible for inappropriately chosen mixing distributions to cause Mixup training to fail to minimize the original empirical risk.

# 2.4 SUFFICIENT CONDITIONS FOR MINIMIZING THE ORIGINAL RISK

The natural follow-up question to the results of the previous subsection is: under what conditions on the data can this failure case be avoided? In other words, when can the Mixup-optimal classifier classify the original data points correctly while being continuous at those points?

Prior to answering that question, we first point out that if discontinuous functions are allowed, then Mixup training always minimizes the original risk on finite datasets:

Proposition 2.8. Consider  $k$ -class classification where the supports  $X_{1}, \ldots, X_{k}$  are finite and  $\mathbb{P}_X$  corresponds to the discrete uniform distribution. Then for every  $h \in \operatorname{argmin}_{g \in \mathcal{C}^{*}} J_{mix}(g, \mathbb{P}_X, \mathbb{P}_f)$ , we have that  $h^i(x) = 1$  on  $X_{i}$ .

Proof Sketch. This is just because the  $(P_X \times P_X \times P_f)$ -measure of mixing a point with itself is constant as  $\epsilon \to 0$ .

Note that Proposition 2.8 holds for any continuous mixing distribution  $\mathbb{P}_f$  supported on  $[0,1]$  - we just need a rich enough model class. In order to obtain the result of Proposition 2.8 with the added restriction of continuity of  $h$  on each of the  $X_{i}$ , we need to make further assumptions. Namely, we

![](images/f62326ffe49c255a210b8363e11b7b0d72496c51bef996fe8f2d279f91029cb5.jpg)  
(a) MNIST

![](images/28a53691ed0b803152348eccc2e76d51d90fd2e5b15bd7f832309fe4025dbd69.jpg)  
Figure 2: Training error plots for Mixup using  $\alpha = 1024$  and regular training on MNIST, CIFAR-10, and CIFAR-100. Each curve above corresponds to the mean of 5 training runs of 50 epochs, and the area around each curve represents a region of one standard deviation. Each of the Mixup models have a final training error that is within  $1\%$  of the non-Mixup models.  
(b) CIFAR-10

![](images/6a2389cf7468d3433d13f51a59ea6dead65e8c5ee429f4e5a4af5105dd628a0b.jpg)  
(c) CIFAR-100

need to avoid the collinearity of different class points that occurred in the previous subsection, and we do so with the following assumption which is a function of a class  $i$  and a point  $x$ :

Assumption 2.9. For a class  $i$  and a point  $x$ , there exists an  $\epsilon > 0$  such that  $A_{x,\epsilon'}^{j,q}$  has measure zero for all  $\epsilon' \leq \epsilon$  when both  $j \neq i$  and  $q \neq i$ .

A visualization of Assumption 2.9 is provided in Section B of the appendix. With this assumption in hand, we obtain the following result as a corollary of Theorem 3.2 which is proved in the next section:

Theorem 2.10. We consider the same setting as Proposition 2.8 and further suppose that Assumption 2.9 is satisfied by all points in  $\bigcup_{i=1}^{k} X_i$ . Then for every  $h \in \operatorname{argmin}_{g \in \mathcal{C}^*} J_{mix}(g, \mathbb{P}_X, \mathbb{P}_f)$ , we have that  $h^i(x) = 1$  on  $X_i$  and that  $h$  is continuous on  $\bigcup_{i=1}^{k} X_i$ .

Application of Sufficient Conditions. Theorem 2.10 suggests a way to test whether a dataset will be amenable to Mixup; we simply attempt to verify if Assumption 2.9 holds for some large enough  $\epsilon$  value (depending on the Lipschitz constant of the model). This is, however, computationally intensive for large, high-dimensional datasets. We thus consider the following approximate verification scheme: we sample an epoch's worth of Mixup points (to simulate training) from a downsampled version of the train dataset, and then compute the minimum distances between each Mixup point and points (from both train and test data) of classes other than the mixed classes. The minimum over these distances corresponds to an estimate of  $\epsilon$  in Assumption 2.9. For our experiments, we consider MNIST, CIFAR-10, and CIFAR-100 (Krizhevsky, 2009) downsampled to  $20\%$  of their sizes (replicating the setting of Guo et al. (2019)) and use angular distance instead of Euclidean distance since ReLU activations are positive homogeneous. We use  $\mathrm{Beta}(\alpha, \alpha)$  with  $\alpha = 1024$  as the mixing distribution both because the results/experiments of Subsection 2.3 demonstrate that the underfitting issue manifests in practice more readily for concentrated distributions and because larger values of  $\alpha$  move further away from the ERM regime.

We find that the  $\epsilon$  value computed according to our scheme is approximately 0.1 or greater (in angular distance) for each of MNIST, CIFAR-10, and CIFAR-100. Given the large Lipschitz constant estimates (Scaman & Virmaux, 2019) for the deep models typically used for these datasets, this much separation between the original points and the mixed points seems to imply that Mixup training should minimize the original risk. We verify this by training ResNet-18 (He et al., 2015) (using the popular implementation of Kuang Liu) on MNIST, CIFAR-10, and CIFAR-100 with and without Mixup for 50 epochs using  $\alpha = 1024$  and a batch size of 128 (with otherwise identical settings to the previous subsection). Results are shown in Figure 2. We checked that the graphs shown are not sensitive to small changes in the aforementioned hyperparameters, although we did not perform an exhaustive hyperparameter search due to resource constraints.

As predicted from our approximate  $\epsilon$  calculation and Theorem 2.10, Mixup training minimizes the empirical risk on MNIST, CIFAR-10, and CIFAR-100. However, we find that the test performance of Mixup at our choice of  $\alpha = 1024$  is significantly worse than ERM for CIFAR-10 and CIFAR-100, affirming what was observed previously by Guo et al. (2019) for choices of  $\alpha >> 1$ . This is in contrast to the fact that the test data points exhibit greater angular distance to the mixed training

points than the original training points do themselves. As such, we challenge the implication in Guo et al. (2019) that collisions between mixed points and test points are the cause of the degradation in test performance - understanding when Mixup generalizes poorly seems to require more than just this perspective.

# 2.5 THE RATE OF EMPIRICAL RISK MINIMIZATION USING MIXUP

Another striking aspect of the experiments in Figure 2 is that Mixup training minimizes the original empirical risk at a very similar rate to that of direct empirical risk minimization. A priori, there is no reason to expect that Mixup should be able to do this - a simple calculation shows that Mixup training only sees one true data point per epoch in expectation (each pair of points is sampled with probability  $\frac{1}{m^2}$  and there are  $m$  true point pairs and  $m$  pairs seen per epoch, where  $m$  is the dataset size). The experimental results are even more surprising given that we are training using  $\alpha = 1024$ , which essentially corresponds to training using the midpoints of the original data points. This seems to imply that it is possible to recover the classifications of the original data points from the midpoints alone (not including the midpoint of a point and itself). We make this rigorous with the following result:

Theorem 2.11. Suppose  $\{x_1, \ldots, x_m\}$  with  $m \geq 6$  are sampled from  $\bigcup_{i=1}^{k} X_i$  according to  $\mathbb{P}_X$ , and that  $\mathbb{P}_X$  has a density (in other words, a continuous distribution). Then with probability 1, we can uniquely recover the points  $\{x_1, \ldots, x_m\}$  given only the  $\binom{m}{2}$  midpoints  $\{x_{i,j}\}_{1 \leq i < j \leq m}$ .

Proof Sketch. The idea is to represent the recovery problem as a linear system, and show using rank arguments that the non-recoverable points are a measure zero set.

Theorem 2.11 shows, in an information-theoretic sense, that it is possible to obtain the original data points (and therefore also their labels) from only their midpoints. While this gives more theoretical backing as to why it is possible for Mixup training using Beta(1024, 1024) to recover the original data point classifications with very low error, it does not explain why this actually happens in practice at the rate that it does. A full theoretical analysis of this phenomenon would necessarily require analyzing the training dynamics of neural networks (or another model of choice) when trained only on midpoints of the original data, which is outside the intended scope of this work. That being said, we hope that such analysis will be a fruitful line of investigation for future work.

# 3 GENERALIZATION PROPERTIES OF MIXUP CLASSIFIERS

Having discussed how Mixup training differs from standard empirical risk minimization with regards to the original training data, we now consider how a learned Mixup classifier can differ from one learned through empirical risk minimization on unseen test data. To do so, we analyze the per-class margin of Mixup classifiers, i.e. the distance one can move from a class support  $X_{i}$  while still being classified as class  $i$ .

# 3.1 THE MARGIN OF MIXUP CLASSIFIERS

Intuitively, if a point  $x$  falls only on line segments between  $X_{i}$  and some other classes  $X_{j}, \ldots$ , and if  $x$  always falls closer to  $X_{i}$  than the other classes, we can expect  $x$  to be classified according to class  $i$  by the Mixup-optimal classifier due to Lemma 2.3. To make this rigorous, we introduce another assumption in the same vein as Assumption 2.9:

Assumption 3.1. For a class  $i$  and a point  $x$ , suppose there exists an  $\epsilon > 0$  and a  $0 < \delta < \frac{\min_j d(X_i, X_j)}{2}$  such that  $A_{x,\epsilon',\delta}^{i,j}$  has measure zero for all  $\epsilon' \leq \epsilon$ .

Here the measure zero condition on the sets  $A_{x,\epsilon',\delta}^{i,\hat{j}}$  is codifying the aforementioned idea that the point  $x$  falls closer to  $X_i$  than any other class on every line segment that intersects it. A visualization of Assumption 3.1 is provided in Section B of the Appendix. Now for points for which Assumptions 2.9 and 3.1 hold, we can prove:

Theorem 3.2. Consider  $k$ -class classification where the supports  $X_{1}, \ldots, X_{k}$  are finite and  $\mathbb{P}_X$  corresponds to the discrete uniform distribution. If a point  $x$  satisfies Assumptions 2.9 and 3.1 with

![](images/47546da7f07a68ae794d2fcabc16bdffaf9bce38f8b2b264f2351cc3bd80955a.jpg)  
Base Model

![](images/b90c48d5fe0856b5af61313283e58f63aca68fca22a1e0cc0328b05053b1804b.jpg)  
Figure 3: Decision boundary plots for standard and Mixup training on the two moons dataset of Pezeshki et al. (2020) with a class separation of 0.5. Each boundary represents the average of 10 training runs of 1500 epochs.  
Mixup Model (alpha = 1)

![](images/2bd8a6f49af920bb41cff64779f12992fd6254437550b3946cb90f1042e970ac.jpg)  
Mixup Model (alpha = 1024)

respect to a class  $i$ , then for every  $h \in \operatorname{argmin}_{g \in \mathcal{C}^*} J_{mix}(g, \mathbb{P}_X, \mathbb{P}_f)$ , we have that  $h$  classifies  $x$  as class  $i$  and that  $h$  is continuous at  $x$ .

Proof Sketch. The limit can be shown to exist using the Lebesgue differentiation theorem, and we can bound the limit below since the  $A_{x,\epsilon',\delta}^{i,j}$  have measure zero.

Any point  $x \in X_i$  is easily seen to satisfy Assumption 3.1 with respect to class  $i$ , and hence we get Theorem 2.10 as a corollary of Theorem 3.2 as mentioned in Section 2. To use Theorem 3.2 to understand generalization, we make the observation that a point  $x$  can satisfy Assumptions 2.9 and 3.1 while being a distance of up to  $\frac{\min_j d(X_i, X_j)}{2}$  from some class  $i$ . This distance can be significantly farther than, for example, the optimal linear separator in a linearly separable dataset.

Experiments. To illustrate this, we consider the two moons dataset (Buitinck et al., 2013), which consists of two classes of points supported on semicircles with added Gaussian noise. Our motivation for doing so comes from the work of Pezeshki et al. (2020), in which it was noted that neural network models trained on a separated version of the two moons dataset essentially learned a linear separator while ignoring the curvature of the class supports. While Pezeshki et al. (2020) introduced an explicit regularizer to encourage a nonlinear decision boundary, we expect due to Theorem 3.2 that Mixup training will achieve a similar result without any additional modifications.

To verify this empirically, we train a two-layer neural network with 500 hidden units with and without Mixup, to have a 1-to-1 comparison with the setting of Pezeshki et al. (2020). We use  $\alpha = 1$  and  $\alpha = 1024$  for Mixup to capture a wide band of mixing densities. The version of the two moons dataset we use is also identical to that of the one used in the experiments of Pezeshki et al. (2020), and we are grateful to the authors for releasing their code under the MIT license. We do full-batch training with all other training, implementation, and compute details remaining the same as the previous section. Results are shown in Figure 3.

Our results affirm the observations of Pezeshki et al. (2020) and previous work (des Combes et al., 2018) that neural network training dynamics may ignore salient features of the dataset; in this case the "Base Model" learns to differentiate the two classes essentially based on the  $x$ -coordinate alone. On the other hand, the models trained using Mixup have highly nonlinear decision boundaries. Further experiments for different class separations and values of  $\alpha$  are included in Section F of the Appendix.

# 3.2 WHEN MIXUP TRAINING LEARNS THE SAME CLASSIFIER

The experiments and theory of the previous sections have shown how a Mixup classifier can differ significantly from one learned through standard training. In this subsection, we now consider the opposing question - when is the Mixup classifier the same as the one learned through standard training? The motivation for doing so is the increasing computational cost of model training; knowing when Mixup produces the same result as ERM allows a practitioner to avoid having to try Mixup training.

Towards that end, we consider the case of binary classification using a linear model  $\theta^{\top}x$  on high-dimensional Gaussian data, which is a setting that arises naturally when training using Gaussian kernels. Specifically, we consider the dataset  $X$  to consist of  $n$  points in  $\mathbb{R}^d$  distributed according to  $\mathcal{N}(0,I_d)$  with  $d > n$  (to be made more precise shortly). We also consider the mixing distribution to

be any symmetric distribution supported on  $[0,1]$  (thereby including as a special case  $\mathrm{Beta}(\alpha ,\alpha)$ ). We let the labels of points in  $X$  be  $\pm 1$  (so that the sign of  $\theta^{\top}x$  is the classification), and use  $X_{1}$  and  $X_{-1}$  to denote the individual class points. We will show that in this setting, the optimal Mixup classifier is the same (up to rescaling of  $\theta$ ) as the ERM classifier learned using gradient descent with high probability. To do so we need some additional definitions.

Definition 3.3. We say  $\hat{\theta}$  is an interpolating solution, if there exists  $k > 0$  such that

$$
\hat {\theta} ^ {\top} x _ {i} = - \hat {\theta} ^ {\top} z _ {j} = k \forall x _ {i} \in X _ {1}, \forall z _ {j} \in X _ {- 1}.
$$

Definition 3.4. The maximum margin solution  $\hat{\theta}$  is defined through:

$$
\tilde{\theta} := \operatorname *{argmax}_{\| \theta \|_{2} = 1}\left\{\min_{x_{i}\in X_{1},z_{j}\in X_{-1}}\left\{\theta^{\top}x_{i}, - \theta^{\top}z_{j}\right\} \right\}
$$

When the maximum margin solution coincides with an interpolating solution for the dataset  $X$  (i.e. all the points are support vectors), we have that Mixup training leads to learning the max margin solution (up to rescaling).

Theorem 3.5. If the maximum margin solution for  $X$  is also an interpolating solution for  $X$ , then any  $\theta$  that lies in the span of  $X$  and minimizes the Mixup loss  $J_{mix}$  for a symmetric mixing distribution  $\mathbb{P}_f$  is a rescaling of the maximum margin solution.

Proof Sketch. It can be shown that  $\theta$  is an interpolating solution using a combination of the strict convexity of  $J_{\text{mix}}$  as a function of  $\theta$  and the symmetry of the mixing distribution.

Remark 3.6. For every  $\theta$ , we can decompose it as  $\theta = \theta_{X} + \theta_{X^{\perp}}$  where  $\theta_{X}$  is the projection of  $\theta$  onto the subspace spanned by  $X$ . By definition we have that  $\theta_{X^{\perp}}$  is orthogonal to all possible mixings of points in  $X$ . Hence,  $\theta_{X^{\perp}}$  does not affect the Mixup loss or the interpolating property, so for simplicity we may just assume  $\theta$  lies in the span of  $X$ .

To characterize the conditions on  $X$  under which the maximum margin solution interpolates the data, we use a key result of Muthukumar et al. (2020), restated below. Note that Muthukumar et al. (2020) actually provide more settings in their paper, but we constrain ourselves to the one stated below for simplicity.

Lemma 3.7. [Theorem 1 in Muthukumar et al. (2020), Rephrased] Assuming  $d > 10n\ln n + n - 1$  then with probability at least  $1 - 2/n$ , the maximum margin solution for  $X$  is also an interpolating solution.

To tie the optimal Mixup classifier back to the classifier learned through standard training, we appeal to the fact that minimizing the empirical cross-entropy of a linear model using gradient descent leads to learning the maximum margin solution on linearly separable data (Soudry et al., 2018; Ji & Telgarsky, 2018). From this we obtain the desired result of this subsection:

Corollary 3.8. Under the same conditions as Lemma 3.7, the optimal Mixup classifier has the same direction as the classifier learned through minimizing the empirical cross-entropy using gradient descent with high probability.

# 4 CONCLUSION

The main contribution of our work has been to provide a theoretical framework for analyzing how Mixup training can differ from empirical risk minimization. Our results characterize a practical failure case of Mixup, and also identify conditions under which Mixup can provably minimize the original risk. They also show in the sense of margin why the generalization of Mixup classifiers can be superior to those learned through empirical risk minimization, while again identifying model classes and datasets for which the generalization of a Mixup classifier is no different (with high probability). We also emphasize that the generality of our theoretical framework allows most of our results to hold for any continuous mixing distribution. Our hope is that the tools developed in this work will see applications in future works concerned with analyzing the relationship between benefits obtained from Mixup training and properties of the training data.

# 5 ETHICS STATEMENT

We do not anticipate any direct misuses of this work due to its theoretical nature. That being said, the failure case of Mixup discussed in Section 2 could serve as a way for an adversary to potentially exploit a model trained using Mixup to classify data incorrectly. However, as this requires knowledge of the mixing distribution and other hyperparameters of the model, we do not flag this as a significant concern - we would just like to point it out for completeness.

# 6 REPRODUCIBILITY STATEMENT

Full proofs for all results in the main body of the paper can be found in Sections C and E of the Appendix. We have also included all of the code used to generate the plots and experimental results in this paper in the supplementary material. We have tried our best to organize the code to be easy to use and extend. Detailed instructions for how to run each type of experiment are provided in the README file included with the code.

# REFERENCES

Eric Arazo, Diego Ortego, Paul Albert, Noel E O'Connor, and Kevin McGuinness. Unsupervised label noise modeling and loss correction. arXiv preprint arXiv:1904.11238, 2019.  
David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf.  
Lars Buitinck, Gilles Loupe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae, Peter Prettenhofer, Alexandre Gramfort, Jaques Grobler, Robert Layton, Jake VanderPlas, Arnaud Joly, Brian Holt, and Gael Varoquaux. API design for machine learning software: experiences from the scikit-learn project. CoRR, abs/1309.0238, 2013. URL http://arxiv.org/abs/1309.0238.  
Luigi Carratino, Moustapha Cisse, Rodolphe Jenatton, and Jean-Philippe Vert. On mixup regularization, 2020.  
Ching-Yao Chuang and Youssef Mroueh. Fair mixup: Fairness via interpolation. CoRR, abs/2103.06503, 2021. URL https://arxiv.org/abs/2103.06503.  
Remi Tachet des Combes, Mohammad Pezeshki, Samira Shabanian, Aaron C. Courville, and Yoshua Bengio. On the learning dynamics of deep neural networks. CoRR, abs/1809.06848, 2018. URL http://arxiv.org/abs/1809.06848.  
Mahyar Fazlyab, Alexander Robey, Hamed Hassani, Manfred Morari, and George J. Pappas. Efficient and accurate estimation of lipschitz constants for deep neural networks, 2019.  
Hongyu Guo, Yongyi Mao, and Richong Zhang. Mixup as locally linear out-of-manifold regularization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 3714-3722, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.  
Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.  
Yangsibo Huang, Zhao Song, Kai Li, and Sanjeev Arora. Instahide: Instance-hiding schemes for private distributed learning, 2021.

Ziwei Ji and Matus Telgarsky. Risk and parameter convergence of logistic regression. CoRR, abs/1803.07300, 2018. URL http://arxiv.org/abs/1803.07300.  
Jang-Hyun Kim, Wonho Choo, and Hyun Oh Song. Puzzle mix: Exploiting saliency and local statistics for optimal mixup. In International Conference on Machine Learning, pp. 5275-5285. PMLR, 2020.  
JangHyun Kim, Wonho Choo, Hosan Jeong, and Hyun Oh Song. Co-mixup: Saliency guided joint mixup with supermodular diversity. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=gvxJzw8kW4b.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun (eds.), 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1412.6980.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.  
Alex Lamb, Vikas Verma, Juho Kannala, and Yoshua Bengio. Interpolated adversarial training: Achieving robust neural networks without sacrificing too much accuracy. In Proceedings of the 12th ACM Workshop on Artificial Intelligence and Security, pp. 95-103, 2019.  
Fabian Latorre, Paul Rolland, and Volkan Cevher. Lipschitz constant estimation of neural networks via sparse polynomial optimization, 2020.  
Olivier Marchal and Julyan Arbel. On the sub-gaussianity of the beta and dirichlet distributions. Electronic Communications in Probability, 22(none), Jan 2017. ISSN 1083-589X. doi: 10.1214/17-ecp92. URL http://dx.doi.org/10.1214/17-ECP92.  
Vidya Muthukumar, Adhyyan Narang, Vignesh Subramanian, Mikhail Belkin, Daniel Hsu, and Anant Sahai. Classification vs regression in overparameterized regimes: Does the loss function matter? arXiv preprint arXiv:2005.08054, 2020.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pp. 8026-8037. Curran Associates, Inc., 2019.  
Mohammad Pezeshki, Sekou-Oumar Kaba, Yoshua Bengio, Aaron C. Courville, Doina Precup, and Guillaume Lajoie. Gradient starvation: A learning proclivity in neural networks. CoRR, abs/2011.09468, 2020. URL https://arxiv.org/abs/2011.09468.  
Kevin Scaman and Aladin Virmaux. Lipschitz regularity of deep neural networks: analysis and efficient estimation, 2019.  
Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 596-608. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/06964dce9addb1c5cb5d6e3d9838f733-Paper.pdf.  
Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. *The Journal of Machine Learning Research*, 19(1): 2822-2878, 2018.  
Sunil Thulasidasan, Gopinath Chennupati, Jeff A Bilmes, Tanmoy Bhattacharya, and Sarah Michalak. On mixup training: Improved calibration and predictive uncertainty for deep neural networks. Advances in Neural Information Processing Systems, 32:13888-13899, 2019.

Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, and Yoshua Bengio. Manifold mixup: Better representations by interpolating hidden states. In International Conference on Machine Learning, pp. 6438-6447. PMLR, 2019.

Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6023-6032, 2019.

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018.

Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani, and James Zou. How does mixup help with robustness and generalization?, 2020.

Linjun Zhang, Zhun Deng, Kenji Kawaguchi, and James Zou. When and how mixup improves calibration, 2021.
